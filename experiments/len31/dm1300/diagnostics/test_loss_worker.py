"""Full held-out cross-entropy for saved checkpoints; independent of training."""
import argparse, csv, fcntl, hashlib, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def now():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)

def fingerprint(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]

def idle_gpu():
    # Only consider GPU 4, never any of the four training GPUs.
    try:
        row = subprocess.check_output(['nvidia-smi', '--id=4', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10).strip().split(',')
        uuid, used = row[0].strip(), int(row[1])
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True, timeout=10)
        return uuid if used <= 64 and uuid not in apps else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

def evaluate(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.run_dir / 'source_at_launch'))
    from model import ReasoningTransformer
    started = time.monotonic()
    cfg = json.loads((args.run_dir / 'config.json').read_text())
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if any(saved['config'].get(k) != v for k, v in cfg.items() if k != 'started_at'):
        raise ValueError('Checkpoint/run configuration mismatch')
    epoch = int(saved['epoch'])
    model = ReasoningTransformer(**cfg['model'])
    model.load_state_dict(saved['model'])
    del saved
    model.to(device=args.device, dtype=torch.float32).eval()
    x = np.load(args.data_dir / 'eval_x.npy', mmap_mode='r')
    y = np.load(args.data_dir / 'eval_y.npy', mmap_mode='r')
    n = cfg['data']['eval_per_group']
    if x.shape != (2*n, cfg['model']['length']) or y.shape != (2*n,):
        raise ValueError('Unexpected evaluation data shape')
    groups = {}
    with torch.inference_mode():
        for name, offset in [('test_canonical', 0), ('test_noncanonical', n)]:
            loss_sum, correct, seen = 0.0, 0, 0
            # Match the training evaluator's rank shards and batch size exactly.
            for rank in range(cfg['gpus']):
                ids = np.arange(rank, n, cfg['gpus']) + offset
                for left in range(0, len(ids), cfg['eval_batch']):
                    selected = ids[left:left+cfg['eval_batch']]
                    tokens = torch.from_numpy(np.array(x[selected], dtype=np.int64)).to(args.device)
                    targets = torch.from_numpy(np.array(y[selected], dtype=np.int64)).to(args.device)
                    logits = model(tokens)
                    losses = F.cross_entropy(logits.float(), targets, reduction='none')
                    if not bool(torch.isfinite(losses).all()):
                        raise ValueError('Nonfinite test loss')
                    loss_sum += losses.double().sum().item()
                    correct += logits.argmax(-1).eq(targets).sum().item()
                    seen += len(selected)
            groups[name] = dict(loss=loss_sum/seen, loss_sum=loss_sum, correct=correct, n=seen, accuracy=correct/seen)
            print(now(), 'epoch', epoch, name, groups[name], flush=True)
    c, nc = groups['test_canonical'], groups['test_noncanonical']
    row = dict(epoch=epoch, test_loss=(c['loss_sum']+nc['loss_sum'])/(c['n']+nc['n']),
               test_canonical_loss=c['loss'], test_noncanonical_loss=nc['loss'],
               test_loss_n=c['n']+nc['n'], groups=groups, device=args.device,
               precision='fp32_tf32_off', evaluated_at=now(), elapsed_seconds=round(time.monotonic()-started, 2))
    with (args.run_dir / 'accuracy.csv').open() as f:
        original = next((r for r in csv.DictReader(f) if int(r['epoch']) == epoch), None)
    row['accuracy_check'] = {}
    if original:
        for name, values in groups.items():
            if original.get(name+'_correct'):
                delta = values['correct'] - int(original[name+'_correct'])
                row['accuracy_check'][name] = dict(logged_correct=int(original[name+'_correct']), recomputed_correct=values['correct'], difference=delta)
                if abs(delta) > 2:
                    raise ValueError('Recomputed accuracy differs from training log: '+str(row['accuracy_check']))
    sha = hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            sha.update(block)
    row['checkpoint_sha256'] = sha.hexdigest()
    save(args.output, row)

def worker(args):
    os.nice(10)
    lock = (ROOT/'test_loss_worker.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Test loss worker already running')
    path = ROOT/'test_loss_history.json'
    state = json.loads(path.read_text()) if path.exists() else dict(run_name=args.run_dir.name, rows=[])
    if state['run_name'] != args.run_dir.name:
        raise ValueError('Test loss history belongs to another run')
    gpu_failed = False
    while True:
        try:
            done = {tuple(r['fingerprint']) for r in state['rows']}
            candidates = [args.run_dir/'latest.pt'] + sorted((args.run_dir/'checkpoints').glob('epoch_*.pt'), reverse=True)
            source = next((p for p in candidates if tuple(fingerprint(p)) not in done), None)
            if source is None:
                state.update(state='ready', updated_at=now(), current_checkpoint=None)
                save(path, state)
                time.sleep(30)
                continue
            pinned = ROOT/'test_loss_checkpoint.pt'
            pinned.unlink(missing_ok=True)
            os.link(source, pinned)
            fp = fingerprint(pinned)
            gpu = None if gpu_failed else idle_gpu()
            device = 'cuda' if gpu else 'cpu'
            state.update(state='evaluating', updated_at=now(), current_checkpoint=source.name, current_device=device)
            save(path, state)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu or '', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NVIDIA_TF32_OVERRIDE='0')
            output = ROOT/'test_loss_result.json'
            cmd = [sys.executable, str(Path(__file__).resolve()), '--evaluate', '--run-dir', str(args.run_dir), '--data-dir', str(args.data_dir), '--checkpoint', str(pinned), '--output', str(output), '--device', device]
            print(now(), 'Evaluating', source.name, 'on', device, flush=True)
            try:
                try:
                    subprocess.run(cmd, env=env, check=True, timeout=2400)
                except subprocess.CalledProcessError:
                    if device != 'cuda':
                        raise
                    gpu_failed = True
                    device = 'cpu'
                    env['CUDA_VISIBLE_DEVICES'] = ''
                    cmd[-1] = device
                    state.update(current_device=device, error=None, updated_at=now())
                    save(path, state)
                    print(now(), 'CUDA evaluation failed; retrying this checkpoint on CPU.', flush=True)
                    subprocess.run(cmd, env=env, check=True, timeout=2400)
                row = json.loads(output.read_text())
                row['fingerprint'] = fp
                state['rows'] = sorted([r for r in state['rows'] if r['epoch'] != row['epoch']] + [row], key=lambda r:r['epoch'])
                state.update(state='ready', updated_at=now(), current_checkpoint=None, error=None)
                save(path, state)
                print(now(), 'Saved epoch', row['epoch'], 'test_loss', row['test_loss'], flush=True)
            finally:
                pinned.unlink(missing_ok=True)
        except Exception as exc:
            state.update(state='error', updated_at=now(), error=type(exc).__name__)
            save(path, state)
            print(now(), repr(exc), flush=True)
            time.sleep(120)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    args = p.parse_args()
    evaluate(args) if args.evaluate else worker(args)
