"""Evaluate strict forward order for the paused dm1300 checkpoint."""
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
RUN = Path('/home/qintian/multi-reasoning/runs/chain_4step_3layer_dm1300_dff2600_len31_6p5m_4x4080_batch4000_compile1_seed2031_eval10000_wd0p15_lr1e-4_fp32_20261027')
CHECKPOINT = RUN / 'paused_for_4layer_len25_20261107/resume.pt'
DATA = Path('/home/qintian/multi-reasoning/data/chain_4step_6p5m_len31_vocab101_eval10000')
os.nice(10)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
started = time.monotonic()

saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
epoch = int(saved['epoch'])
assert epoch == 1575
config = saved['config']
spec = importlib.util.spec_from_file_location('saved_model', RUN / 'source_at_launch/model.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.ReasoningTransformer(**config['model']).eval()
model.load_state_dict(saved['model'], strict=True)
del saved
x = np.load(DATA / 'eval_x.npy', mmap_mode='r')
y = np.load(DATA / 'eval_y.npy', mmap_mode='r')
assert x.shape == (20000, 31) and y.shape == (20000,)

positions = np.empty((len(x), 4), dtype=np.int64)
for row, tokens in enumerate(x):
    facts = tokens[:-1].reshape(15, 2)
    value = int(tokens[-1])
    for step in range(4):
        hits = np.flatnonzero(facts[:, 0] == value)
        assert len(hits) == 1
        positions[row, step] = hits[0]
        value = int(facts[hits[0], 1])
    assert value == int(y[row])
forward_mask = np.all(np.diff(positions, axis=1) > 0, axis=1)
ids = np.flatnonzero(forward_mask)
assert len(ids) > 0 and not np.any(forward_mask[:10000])
np.save(ROOT / 'existing_forward_sample_ids.npy', ids)

natural_x = np.empty((10000, 31), dtype=np.uint8)
natural_y = np.array(y[:10000], copy=True)
already_fully_sorted = 0
for row, tokens in enumerate(x[:10000]):
    facts = tokens[:-1].reshape(15, 2)
    starts = set(map(int, facts[:, 0])) - set(map(int, facts[:, 1]))
    assert len(starts) == 1
    value = starts.pop()
    order = []
    nodes = [value]
    for step in range(15):
        hits = np.flatnonzero(facts[:, 0] == value)
        assert len(hits) == 1
        order.append(int(hits[0]))
        value = int(facts[hits[0], 1])
        nodes.append(value)
    assert len(set(nodes)) == 16
    natural_x[row, :-1] = facts[order].reshape(-1)
    natural_x[row, -1] = tokens[-1]
    query_position = nodes.index(int(tokens[-1]))
    assert nodes[query_position + 4] == int(natural_y[row])
    assert np.array_equal(np.sort(natural_x[row, :-1].reshape(15, 2), axis=0),
                          np.sort(facts, axis=0))
    assert np.all(np.isin((facts[:, 1].astype(int) - facts[:, 0]) % 5, [2, 3]))
    already_fully_sorted += int(order == list(range(15)))
    paired = x[row + 10000, :-1].reshape(15, 2)
    assert sorted(map(tuple, paired.tolist())) == sorted(map(tuple, facts.tolist()))
    assert int(x[row + 10000, -1]) == int(tokens[-1])
    assert int(y[row + 10000]) == int(natural_y[row])

def evaluate(name, tokens, targets):
    predictions = []
    loss_sum = 0.0
    with torch.inference_mode():
        for start in range(0, len(tokens), 250):
            batch_x = torch.from_numpy(np.array(tokens[start:start+250], dtype=np.int64))
            batch_y = torch.from_numpy(np.array(targets[start:start+250], dtype=np.int64))
            logits = model(batch_x)
            predictions.append(logits.argmax(-1).numpy())
            loss_sum += F.cross_entropy(logits.float(), batch_y, reduction='sum').item()
            if (start // 250 + 1) % 10 == 0:
                print(name, min(start+250, len(tokens)), '/', len(tokens), flush=True)
    predicted = np.concatenate(predictions)
    correct = int((predicted == targets).sum())
    result = dict(n=len(targets), correct=correct, accuracy=correct/len(targets),
                  cross_entropy=loss_sum/len(targets))
    np.save(ROOT / (name + '_predictions.npy'), predicted)
    print(name, json.dumps(result), flush=True)
    return result

result = dict(epoch=epoch, checkpoint=str(CHECKPOINT), device='cpu',
              precision='fp32_tf32_off', definitions={
                  'existing_forward_subset':'Original test examples with p(F1)<p(F2)<p(F3)<p(F4); other facts stay unchanged',
                  'full_chain_forward':'All 15 facts reordered along the complete chain, original query and answer unchanged; 10000 unique paired tasks counted once'},
              existing_subset_counts={'canonical':int(forward_mask[:10000].sum()),
                                      'noncanonical':int(forward_mask[10000:].sum())},
              original_can_already_fully_sorted=already_fully_sorted)
result['existing_forward_subset'] = evaluate('existing_forward_subset', x[ids], y[ids])
(ROOT / 'result_partial.json').write_text(json.dumps(result, indent=2) + '\n')
result['full_chain_forward'] = evaluate('full_chain_forward', natural_x, natural_y)
with (RUN/'accuracy.csv').open() as f:
    baseline = next(row for row in csv.DictReader(f) if int(row['epoch']) == epoch)
result['logged_original_test_accuracy'] = {k:float(baseline[k]) for k in
    ['test_accuracy','test_canonical_accuracy','test_noncanonical_accuracy']}
result['checkpoint_sha256'] = hashlib.file_digest(CHECKPOINT.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
result['elapsed_seconds'] = round(time.monotonic() - started, 2)
(ROOT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('COMPLETE', json.dumps(result), flush=True)