"""Live full-history plots for the three-layer, four-step reasoning runs.

Overall test accuracy is reweighted to the natural order distribution:
canonical / 3 + noncanonical * 2 / 3. Raw paired evaluation counts are retained.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

WEIGHTED_KEY = 'test_accuracy_natural_weighted'
FORMULA = 'canonical / 3 + noncanonical * 2 / 3'


def atomic_bytes(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(contents)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def weighted_rows(rows):
    result = []
    for original in rows:
        row = dict(original)
        row[WEIGHTED_KEY] = ''
        can, non = row.get('test_canonical_accuracy'), row.get('test_noncanonical_accuracy')
        if can not in (None, '') and non not in (None, ''):
            can, non = float(can), float(non)
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (can, non)):
                raise ValueError('Invalid conditional accuracy')
            row[WEIGHTED_KEY] = can/3 + 2*non/3
        result.append(row)
    return result


def render(csv_path: Path, png_path: Path, steps: int | None = None):
    csv_path, png_path = Path(csv_path), Path(png_path)
    raw = csv_path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    fields = list(reader.fieldnames or [])
    rows = weighted_rows(list(reader))
    if not rows:
        return None
    config_path = csv_path.parent/'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    steps = steps or config.get('data', {}).get('steps')
    if steps not in (None, 4):
        raise ValueError('The 1:2 weighting is configured for these four-step experiments')
    tests = [r for r in rows if r[WEIGHTED_KEY] != '']
    latest = tests[-1] if tests else None
    names = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ('Microsoft YaHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
                                   'WenQuanYi Micro Hei', 'Droid Sans Fallback') if name in names), 'DejaVu Sans')
    plt.rcParams.update({'font.family':font, 'font.size':12, 'axes.unicode_minus':False,
                         'axes.spines.top':False, 'axes.spines.right':False})
    series = [('train_accuracy','训练集（固定 10,000 条）','#2471b5'),
              (WEIGHTED_KEY,'整体测试（1:2 加权）','#6b7280'),
              ('test_canonical_accuracy','Canonical 测试集','#d35b19'),
              ('test_noncanonical_accuracy','Noncanonical 测试集','#19856b')]
    fig, ax = plt.subplots(figsize=(14,7.4))
    try:
        fig.subplots_adjust(left=.075,right=.97,top=.72,bottom=.19)
        for key, label, color in series:
            points = [r for r in rows if r.get(key) not in (None, '')]
            if points:
                ax.plot([int(r['epoch']) for r in points], [100*float(r[key]) for r in points],
                        label=label, color=color, linewidth=1.8 if key=='test_canonical_accuracy' else 1.5)
                if key != 'train_accuracy':
                    ax.scatter([int(points[-1]['epoch'])],[100*float(points[-1][key])],color=color,s=24,zorder=5)
        epoch = int(rows[-1]['epoch'])
        ax.set(xlim=(0,max(1,epoch*1.01)),ylim=(0,102),xlabel='Epoch',ylabel='准确率（%）')
        ax.set_yticks(range(0,101,10)); ax.grid(alpha=.2); ax.set_axisbelow(True)
        model, data = config.get('model', {}), config.get('data', {})
        title = 'dm'+str(model['width']) if 'width' in model else '模型'
        if data.get('train_size'):
            title += f' · {data["train_size"]/10000:g} 万条'
        fig.suptitle(title+' · 完整准确率曲线',fontsize=21,fontweight='bold',y=.96)
        test_epoch = latest['epoch'] if latest else '尚未评估'
        fig.text(.5,.895,f'已完成 epoch {epoch} ｜ 最新测试 epoch {test_epoch} ｜ 从第 0 轮开始',
                 ha='center',fontsize=12,color='#555555')
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.85),
                   ncol=4,fontsize=11,frameon=False)
        if latest:
            fig.text(.5,.785,
                f'最新：训练抽样 {100*float(rows[-1]["train_accuracy"]):.2f}% · Canonical {100*float(latest["test_canonical_accuracy"]):.2f}% · Noncanonical {100*float(latest["test_noncanonical_accuracy"]):.2f}% · 整体测试 {100*float(latest[WEIGHTED_KEY]):.2f}%',
                ha='center',fontsize=11,color='#374b5b')
        now = datetime.now(timezone(timedelta(hours=8)))
        fig.text(.075,.105,'灰线：整体测试准确率 = Canonical × 1/3 + Noncanonical × 2/3（自然顺序比例重加权）',fontsize=10.5,color='#374b5b')
        fig.text(.075,.067,'两组测试各 10,000 条；训练为固定 10,000 条抽样。连接实际评估点，未做平滑。',fontsize=10,color='#555555')
        fig.text(.075,.03,'自动更新于 '+now.strftime('%Y-%m-%d %H:%M:%S')+'（香港时间）',fontsize=10,color='#555555')
        buffer = io.BytesIO()
        fig.savefig(buffer,dpi=170,format='png',facecolor='white')
        image = buffer.getvalue()
    finally:
        plt.close(fig)
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output,fieldnames=[f for f in fields if f != WEIGHTED_KEY]+[WEIGHTED_KEY],lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    atomic_bytes(csv_path.parent/'accuracy_reweighted.csv',output.getvalue().encode('utf-8'))
    atomic_bytes(png_path,image)
    if png_path.name == 'accuracy.png':
        atomic_bytes(png_path.parent/'accuracy_natural.png',image)
    record = dict(updated_at=now.isoformat(),epoch=epoch,latest_test_epoch=int(latest['epoch']) if latest else None,
                  overall_test_accuracy=float(latest[WEIGHTED_KEY]) if latest else None,
                  canonical_accuracy=float(latest['test_canonical_accuracy']) if latest else None,
                  noncanonical_accuracy=float(latest['test_noncanonical_accuracy']) if latest else None,
                  formula=FORMULA,raw_csv_sha256=hashlib.sha256(raw).hexdigest(),
                  image_sha256=hashlib.sha256(image).hexdigest(),image_name=png_path.name,
                  test_points=len(tests),pid=os.getpid())
    atomic_bytes(csv_path.parent/'accuracy_plot_status.json',(json.dumps(record,indent=2)+'\n').encode())
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv',type=Path)
    parser.add_argument('png',type=Path)
    parser.add_argument('--steps',type=int)
    args = parser.parse_args()
    render(args.csv,args.png,args.steps)
