"""Update cached attention only when an atomically saved checkpoint changes.
Runs inference on CPU with two threads and low scheduling priority.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def now():
    return datetime.now(timezone.utc).isoformat()

def fingerprint(path):
    s=path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]

def write_manifest(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--poll-seconds',type=float,default=30)
    p.add_argument('--once',action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    cache=root/'attention_cache';cache.mkdir(exist_ok=True)
    lock=(root/'attention_worker.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another attention worker is already running',flush=True)
        return
    manifest_path=root/'attention_manifest.json'
    previous={}
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
    last=previous.get('latest')
    try:
        os.nice(10)
    except OSError:
        pass
    while True:
        checkpoint=args.run_dir/'latest.pt'
        try:
            observed=fingerprint(checkpoint)
            if last and last.get('fingerprint')==observed and last.get('view_version')==2 and last.get('color_scale_version')==3:
                if previous.get('state')!='ready':
                    previous=dict(state='ready',updated_at=now(),latest=last)
                    write_manifest(manifest_path,previous)
                if args.once: return
                time.sleep(args.poll_seconds)
                continue
            started=time.monotonic()
            previous=dict(state='building',updated_at=now(),build_started_at=now(),latest=last)
            write_manifest(manifest_path,previous)
            # The trainer uses os.replace; this hard link pins one immutable inode.
            pinned=root/'attention_checkpoint.pt'
            pinned.unlink(missing_ok=True)
            os.link(checkpoint,pinned)
            actual=fingerprint(pinned)
            generation='snapshot_'+str(actual[1])+'_'+str(actual[3])+'_single'
            destination=cache/generation
            work=cache/(generation+'_work')
            if work.exists():
                # Only clean this worker's exact staging directory inside its cache.
                assert work.resolve().parent==cache.resolve() and work.name.endswith('_work')
                shutil.rmtree(work)
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,str(root/'attention_probe.py'),'--repo',str(args.repo),
                 '--run',str(args.run_dir),'--out',str(work),'--checkpoint',str(pinned),
                 '--device','cpu','--n','256','--dashboard']
            print(now(),'Generating attention for',generation,flush=True)
            try:
                with (root/'attention_probe.log').open('w') as log:
                    result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                if result.returncode:
                    raise RuntimeError('Attention probe failed; see attention_probe.log')
                m=json.loads((work/'metrics.json').read_text())
                for name in [f'attention_single_{group}_{layer}.png' for group in ['can','noncan'] for layer in ['all','layer1','layer2','layer3']]:
                    if (work/name).read_bytes()[:8]!=b'\x89PNG\r\n\x1a\n':
                        raise ValueError('Invalid attention image')
                if destination.exists():
                    assert destination.resolve().parent==cache.resolve()
                    shutil.rmtree(destination)
                os.replace(work,destination)
                groups=[m['groups'][n] for n in ['can','noncan']]
                last=dict(epoch=m['epoch'],generation=generation,generated_at=now(),
                          view_version=2,color_scale_version=m['color_scale_version'],
                          color_scale=m['color_scale'],examples=m['examples'],
                          elapsed_seconds=round(time.monotonic()-started,2),
                          fingerprint=actual,checkpoint_sha256=m['checkpoint_sha256'],
                          sample_count=sum(g['n'] for g in groups),sampling_seed=m['sampling_seed'],
                          device='cpu',reconstruction_max_logit_error=m['reconstruction_max_logit_error'],
                          pair_mass=[sum(g['destination_to_own_source_mass'][i] for g in groups)/2 for i in range(3)],
                          pair_top1=[sum(g['destination_to_own_source_top1'][i] for g in groups)/2 for i in range(3)],
                          query_answer_top1=[sum(g['query_to_path_top1']['F4_dst'][i] for g in groups)/2 for i in range(3)])
                previous=dict(state='ready',updated_at=now(),latest=last)
                write_manifest(manifest_path,previous)
                print(now(),'Ready: epoch',last['epoch'],'in',last['elapsed_seconds'],'seconds',flush=True)
            finally:
                pinned.unlink(missing_ok=True)
            if args.once: return
        except Exception as exc:
            previous=dict(state='error',updated_at=now(),latest=last,
                          message='热图暂未更新，稍后自动重试。',error_type=type(exc).__name__)
            write_manifest(manifest_path,previous)
            print(now(),repr(exc),flush=True)
            if args.once: raise
            time.sleep(300)
            continue
        time.sleep(args.poll_seconds)

if __name__=='__main__':
    main()
