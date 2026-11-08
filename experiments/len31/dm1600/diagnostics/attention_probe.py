"""Read-only checkpoint attention probe; writes only to a fresh diagnostics folder."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import torch


def recover_paths(x):
    pairs = x[:, :30].reshape(-1, 15, 2)
    value = x[:, 30].copy()
    path = np.empty((len(x), 4), dtype=int)
    for step in range(4):
        hits = pairs[:, :, 0] == value[:, None]
        assert np.all(hits.sum(1) == 1)
        path[:, step] = hits.argmax(1)
        value = pairs[np.arange(len(x)), path[:, step], 1]
    return path, value


def manual_forward(model, tokens):
    x = model.token(tokens) + model.position(torch.arange(31, device=tokens.device))
    for block in model.blocks:
        normed = block.norm_attention(x)
        q, k, v = block.qkv(normed).view(len(x), 31, 3, -1).unbind(2)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        score = score.masked_fill(torch.ones(31, 31, dtype=torch.bool, device=q.device).triu(1), -torch.inf)
        a = score.softmax(-1)
        y = x + block.out(a @ v)
        x = y + block.ffn(block.norm_ffn(y))
    return model.head(model.final_norm(x[:, -1]))


def edge_stats(a, rows, cols):
    if len(rows) == 0:
        return {'count': 0}
    b, q, k = np.asarray(rows), np.asarray(cols[0]), np.asarray(cols[1])
    weights = a[b, :, q, k]
    top = a.argmax(-1)[b, :, q] == k[:, None]
    return {'count': len(b), 'mass': weights.mean(0).tolist(),
            'top1': top.mean(0).tolist()}


def metrics(x, a, pred, target):
    n = len(x)
    dest = np.arange(1, 30, 2)
    src = dest - 1
    argmax = a.argmax(-1)
    path, answer = recover_paths(x)
    assert np.array_equal(answer, target)
    result = {
        'n': n, 'probe_accuracy': float((pred == target).mean()),
        'destination_to_own_source_mass': a[:, :, dest, src].mean((0, 2)).tolist(),
        'destination_to_own_source_top1': (argmax[:, :, dest] == src).mean((0, 2)).tolist(),
        'destination_to_own_source_mass_excluding_first_fact': a[:, :, dest[1:], src[1:]].mean((0, 2)).tolist(),
        'destination_to_own_source_top1_excluding_first_fact': (argmax[:, :, dest[1:]] == src[1:]).mean((0, 2)).tolist(),
        'pair_mass_per_slot': a[:, :, dest, src].mean(0).tolist(),
        'query_self_mass': a[:, :, 30, 30].mean(0).tolist(),
        'query_to_path_mass': {}, 'query_to_path_top1': {},
    }
    for step in range(4):
        for role, offset in [('src', 0), ('dst', 1)]:
            idx = 2 * path[:, step] + offset
            name = f'F{step+1}_{role}'
            result['query_to_path_mass'][name] = a[np.arange(n), :, 30, idx].mean(0).tolist()
            result['query_to_path_top1'][name] = (argmax[:, :, 30] == idx[:, None]).mean(0).tolist()
    groups = {name: ([], [], []) for name in [
        'source_to_previous_fact_destination', 'destination_to_previous_fact_destination',
        'destination_to_next_fact_destination']}
    pairs = x[:, :30].reshape(n, 15, 2)
    for b in range(n):
        for j, (s, t) in enumerate(pairs[b]):
            previous = np.flatnonzero(pairs[b, :, 1] == s)
            following = np.flatnonzero(pairs[b, :, 0] == t)
            for name, found, row in [
                ('source_to_previous_fact_destination', previous, 2*j),
                ('destination_to_previous_fact_destination', previous, 2*j+1),
                ('destination_to_next_fact_destination', following, 2*j+1),
            ]:
                if len(found) and 2*int(found[0])+1 <= row:
                    bs, qs, ks = groups[name]
                    bs.append(b); qs.append(row); ks.append(2*int(found[0])+1)
    for name, (bs, qs, ks) in groups.items():
        result[name] = edge_stats(a, bs, (qs, ks))
    return result


def attention_colorbar(fig, im, ax, shrink):
    ticks = [0, .05, .15, .3, .5, .75, 1]
    bar = fig.colorbar(im, ax=ax, shrink=shrink, ticks=ticks,
                       label='Attention probability (%) | continuous scale')
    bar.ax.set_yticklabels(['0', '5', '15', '30', '50', '75', '100'])
    bar.ax.minorticks_off()
    return bar


def panel(ax, a, title, labels=None):
    masked = np.ma.masked_array(a, mask=np.triu(np.ones((31, 31), bool), 1))
    cmap = plt.get_cmap('Blues', 1024).copy()
    cmap.set_bad('#d8dde5')
    # A smooth square-root scale keeps small differences visible without hard bins.
    norm = PowerNorm(gamma=.5, vmin=0, vmax=1)
    im = ax.imshow(masked, norm=norm, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel('Key position (attended to)', fontsize=10)
    ax.set_ylabel('Query position (attending from)', fontsize=10)
    ticks = np.arange(31) if labels is not None else np.r_[np.arange(0, 30, 2), 30]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ticklabels = labels if labels is not None else [str(p+1) if p < 30 else '31 Q' for p in ticks]
    ax.set_xticklabels(ticklabels, rotation=90, fontsize=6.5 if labels is not None else 8)
    ax.set_yticklabels(ticklabels, fontsize=6.5 if labels is not None else 8)
    ax.axhline(29.5, color='#25c4bd', lw=.7)
    ax.axvline(29.5, color='#25c4bd', lw=.7)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n', type=int, default=256)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dashboard', action='store_true')
    args = p.parse_args()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(0)
    model_path = args.run / 'source_at_launch/model.py'
    spec = importlib.util.spec_from_file_location('checkpoint_model', model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    checkpoint_path = args.checkpoint or args.run / 'latest.pt'
    with checkpoint_path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
        if digest is None:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
            digest = h.hexdigest()
        f.seek(0)
        saved = torch.load(f, map_location='cpu', weights_only=False)
    epoch, cfg = saved['epoch'], saved['config']
    model = mod.ReasoningTransformer(**cfg['model']).eval().to(device)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    print(f'Loaded epoch {epoch}; {device} diagnostic; checkpoint sha256={digest}', flush=True)
    assert cfg['model']['normalization'] == 'prenorm'
    assert cfg['precision'] == 'fp32'
    data = args.repo / 'data/chain_4step_6p5m_len31_vocab101_eval10000'
    tx, ty = [np.load(data / f'train_{v}.npy', mmap_mode='r') for v in ('x', 'y')]
    ex, ey = [np.load(data / f'eval_{v}.npy', mmap_mode='r') for v in ('x', 'y')]
    rng = np.random.default_rng(20261022)
    ids = rng.choice(10000, args.n, replace=False)
    tids = rng.choice(len(tx), args.n, replace=False)
    datasets = {'train': (tx[tids], ty[tids], tids),
                'can': (ex[ids], ey[ids], ids),
                'noncan': (ex[ids+10000], ey[ids+10000], ids+10000)}
    if args.dashboard:
        datasets.pop('train')
        datasets = {name: tuple(v[:1] for v in values) for name, values in datasets.items()}
    captured = []
    def capture(module, inputs, output):
        q, k, _ = output.view(len(output), 31, 3, -1).unbind(2)
        score = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        score.masked_fill_(torch.ones(31, 31, dtype=torch.bool, device=q.device).triu(1), -torch.inf)
        captured.append(score.softmax(-1).cpu().numpy())
    handles = [block.qkv.register_forward_hook(capture) for block in model.blocks]
    result = {'epoch': epoch, 'checkpoint_sha256': digest, 'checkpoint_path': str(checkpoint_path), 'run': str(args.run),
              'config': cfg, 'sampling_seed': 20261022, 'sample_selection': 'random, before inspecting predictions',
              'device': str(device), 'position_indexing': '1-based: source odd 1..29, destination even 2..30; query 31',
              'groups': {}}
    arrays, predictions = {}, {}
    with torch.inference_mode():
        for name, (x, y, sample_ids) in datasets.items():
            attention, pred = [], []
            for start in range(0, len(x), 16):
                captured.clear()
                logits = model(torch.from_numpy(x[start:start+16].astype(np.int64)).to(device))
                assert len(captured) == 3
                attention.append(np.stack(captured, axis=1))
                pred.append(logits.argmax(-1).cpu().numpy())
            a, prediction = np.concatenate(attention), np.concatenate(pred)
            assert np.allclose(a.sum(-1), 1, atol=2e-6)
            assert np.all(a[:, :, np.triu_indices(31, 1)[0], np.triu_indices(31, 1)[1]] == 0)
            arrays[name], predictions[name] = a, prediction
            result['groups'][name] = metrics(x, a, prediction, y)
            result['groups'][name]['sample_ids'] = sample_ids.tolist()
            print(f'{name}: n={len(x)}, probe accuracy={(prediction==y).mean():.4f}, pairing={result["groups"][name]["destination_to_own_source_mass"]}', flush=True)
        for handle in handles:
            handle.remove()
        check_tokens = torch.from_numpy(datasets['can'][0][:4].astype(np.int64)).to(device)
        official, manual = model(check_tokens), manual_forward(model, check_tokens)
        result['reconstruction_max_logit_error'] = float((official-manual).abs().max())
        torch.testing.assert_close(official, manual, rtol=2e-4, atol=2e-4)
    if args.dashboard:
        examples = {}
        for name in ['can', 'noncan']:
            x, y, sample_ids = datasets[name]
            path, _ = recover_paths(x)
            labels = [f'{j+1}:{int(t)}' for j, t in enumerate(x[0])]
            for step, slot in enumerate(path[0]):
                labels[2*slot] += f' F{step+1}s'
                labels[2*slot+1] += f' F{step+1}t'
            labels[-1] += ' Q'
            examples[name] = dict(id=int(sample_ids[0]), tokens=x[0].tolist(),
                                  path_slots_1based=(path[0]+1).tolist(), answer=int(y[0]),
                                  prediction=int(predictions[name][0]))
            for choice in ['all', 'layer1', 'layer2', 'layer3']:
                layers = range(3) if choice == 'all' else [int(choice[-1])-1]
                fig, axes = plt.subplots(1, len(layers), figsize=(22, 8) if choice == 'all' else (9, 8.3), constrained_layout=True, squeeze=False)
                for ax, layer in zip(axes[0], layers):
                    im = panel(ax, arrays[name][0, layer], f'Layer {layer+1}', labels)
                    for slot in path[0]:
                        ax.plot(2*slot+1, 30, marker='s', markersize=9, markerfacecolor='none', markeredgecolor='#28e0c8', markeredgewidth=1)
                attention_colorbar(fig, im, ax=list(axes[0]), shrink=.72)
                fig.suptitle(f'Epoch {epoch} | {name.upper()} sample #{int(sample_ids[0])} | single sequence\nAnswer={int(y[0])}, prediction={int(predictions[name][0])} | ticks: position:token; F1-F4: reasoning path', fontsize=13)
                fig.savefig(args.out / f'attention_single_{name}_{choice}.png', dpi=160)
                plt.close(fig)
        result['examples'] = examples
        result['view_version'] = 2
        result['color_scale_version'] = 3
        result['color_scale'] = 'continuous Blues; sqrt(p); fixed range 0..1; 15% labeled'
        result['sample_selection'] = 'One fixed preselected test sequence per order; no attention averaging'
        (args.out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'epoch':epoch, 'examples':examples, 'output':str(args.out)}), flush=True)
        return
    mean_test = (arrays['can'].mean(0) + arrays['noncan'].mean(0)) / 2
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.6), constrained_layout=True)
    for layer in range(3):
        im = panel(axs[layer], mean_test[layer], f'Layer {layer+1} | mean held-out attention')
    attention_colorbar(fig, im, ax=axs, shrink=.72)
    fig.suptitle(f'FP32 checkpoint epoch {epoch} | {args.n} canonical + {args.n} paired noncanonical samples\n'
                 'Positions: (1,2), (3,4), ... = (source, destination); position 31 = final query', fontsize=14)
    fig.savefig(args.out / 'attention_3layers_mean.png', dpi=180)
    plt.close(fig)
    for layer in range(3):
        fig, ax = plt.subplots(figsize=(8, 7.3), constrained_layout=True)
        im = panel(ax, mean_test[layer], f'Layer {layer+1} | epoch {epoch} | mean of {2*args.n} held-out samples')
        attention_colorbar(fig, im, ax=ax, shrink=.8)
        fig.savefig(args.out / f'attention_layer{layer+1}.png', dpi=160)
        plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(23, 16), constrained_layout=True)
    examples = {}
    for row, name in enumerate(['can', 'noncan']):
        x, y, sample_ids = datasets[name]
        path, _ = recover_paths(x[:1])
        labels = [f'{j+1}:{int(t)}' for j, t in enumerate(x[0])]
        for step, slot in enumerate(path[0]):
            labels[2*slot] += f' F{step+1}s'
            labels[2*slot+1] += f' F{step+1}t'
        labels[-1] += ' Q'
        examples[name] = {'id': int(sample_ids[0]), 'tokens': x[0].tolist(), 'path_slots_1based': (path[0]+1).tolist(),
                          'answer': int(y[0]), 'prediction': int(predictions[name][0])}
        for layer in range(3):
            im = panel(axs[row, layer], arrays[name][0, layer],
                       f'{name.upper()} | layer {layer+1} | answer={int(y[0])}, pred={int(predictions[name][0])}', labels)
            for step, slot in enumerate(path[0]):
                axs[row, layer].plot(2*slot+1, 30, marker='s', markersize=9,
                                    markerfacecolor='none', markeredgecolor='#28e0c8', markeredgewidth=1)
    attention_colorbar(fig, im, ax=axs, shrink=.6)
    fig.suptitle(f'Epoch {epoch} | one preselected paired example, different fact order\n'
                 'Tick: position:token; F1..F4 = true reasoning path, s/source, t/target; cyan boxes = path target keys in final query row', fontsize=15)
    fig.savefig(args.out / 'attention_3layers_paired_example.png', dpi=160)
    plt.close(fig)
    result['examples'] = examples
    if not args.dashboard:
        np.savez_compressed(args.out / 'attention_arrays.npz',
                            **{f'{k}_attention': v for k, v in arrays.items()},
                            **{f'{k}_tokens': v[0] for k, v in datasets.items()},
                            **{f'{k}_predictions': v for k, v in predictions.items()})
    (args.out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'epoch': epoch, 'reconstruction_max_logit_error': result['reconstruction_max_logit_error'],
                      'examples': examples, 'output': str(args.out)}), flush=True)


if __name__ == '__main__':
    main()
