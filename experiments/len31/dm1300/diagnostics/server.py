"""Read-only live training metrics; no GPU dependencies."""
import argparse, csv, io, json, time, re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

def read_json(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            if attempt == 2: raise
            time.sleep(0.05)

def metrics(run):
    history = []
    for row in csv.DictReader(io.StringIO((run / 'accuracy.csv').read_text())):
        if None in row or any(v is None for v in row.values()): continue
        try:
            parsed = {k: float(v) if v else None for k, v in row.items()}
            if parsed['epoch'] is None or parsed['train_accuracy'] is None: continue
            parsed['epoch'] = int(parsed['epoch'])
            history.append(parsed)
        except (ValueError, TypeError): continue
    if not history: raise ValueError('No accuracy records')
    loss_path = Path(__file__).with_name('test_loss_history.json')
    test_loss = dict(state='waiting', rows=[])
    if loss_path.exists():
        try:
            candidate = read_json(loss_path)
            if candidate.get('run_name') == run.name:
                test_loss = candidate
        except (OSError, ValueError):
            test_loss = dict(state='error', rows=[])
    loss_rows = {int(r['epoch']): r for r in test_loss.get('rows', [])}
    for row in history:
        loss = loss_rows.get(row['epoch'], {})
        for key in ('test_loss', 'test_canonical_loss', 'test_noncanonical_loss', 'test_loss_n'):
            row[key] = loss.get(key)
    status = read_json(run / 'status.json')
    progress = None
    with (run / 'train.log').open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 262144))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        if line.startswith('PROGRESS '):
            try:
                progress = json.loads(line[len('PROGRESS '):])
                break
            except json.JSONDecodeError: continue
    tests = [r for r in history if r['test_accuracy'] is not None]
    now = datetime.now(timezone.utc)
    stamps = [status.get('updated_at'), (progress or {}).get('updated_at')]
    times = [datetime.fromisoformat(s).astimezone(timezone.utc) for s in stamps if s]
    activity = max(times) if times else None
    config = read_json(run / 'config.json')
    return dict(run_name=run.name, fetched_at=now.isoformat(),
        last_activity=activity.isoformat() if activity else None,
        activity_age_seconds=(now-activity).total_seconds() if activity else None,
        status=status, progress=progress, history=history,
        test_loss_status={k:test_loss.get(k) for k in ['state','updated_at','current_checkpoint','error']},
        latest_test_loss=max(loss_rows.values(), key=lambda r:r['epoch']) if loss_rows else None,
        latest_train=history[-1], latest_test=tests[-1] if tests else None,
        best_test=max(tests, key=lambda r: r['test_accuracy']) if tests else None,
        config={k:config.get(k) for k in ['seed','epochs','global_batch','learning_rate','weight_decay','eval_every','train_accuracy_sample_size','model','gpus','precision']})


def attention_state(run):
    root=Path(__file__).resolve().parent
    manifest=root/'attention_manifest.json'
    if not manifest.exists():
        return dict(state='waiting',latest=None)
    result=read_json(manifest)
    latest=result.get('latest')
    if latest:
        generation=latest['generation']
        if not re.fullmatch(r'snapshot_\d+_\d+(?:_single)?',generation):
            raise ValueError('Invalid attention generation')
        stat=(run/'latest.pt').stat()
        current=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]
        if result['state']=='ready' and latest.get('fingerprint')!=current:
            result['state']='queued'
        visible={k:v for k,v in latest.items() if k not in {'fingerprint'}}
        visible['images']={group+'_'+layer:'/attention/'+generation+f'/attention_single_{group}_{layer}.png'
                           for group in ['can','noncan'] for layer in ['all','layer1','layer2','layer3']}
        result['latest']=visible
    return result


def make_handler(run):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, mime, body):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            if self.command != 'HEAD': self.wfile.write(body)
        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == '/':
                    self.send(200, 'text/html; charset=utf-8', Path(__file__).with_name('index.html').read_bytes())
                elif path == '/api/metrics':
                    self.send(200, 'application/json; charset=utf-8', json.dumps(metrics(run),ensure_ascii=False,allow_nan=False).encode())
                elif path == '/api/attention':
                    self.send(200, 'application/json; charset=utf-8', json.dumps(attention_state(run),ensure_ascii=False,allow_nan=False).encode())
                elif re.fullmatch(r'/attention/(snapshot_\d+_\d+(?:_single)?)/(attention_(?:single_(?:can|noncan)_(?:all|layer[123])|3layers_mean|3layers_paired_example|layer[123])\.png)',path):
                    parts=path.split('/')
                    image=Path(__file__).resolve().parent/'attention_cache'/parts[2]/parts[3]
                    if image.is_file():
                        self.send(200,'image/png',image.read_bytes())
                    else:
                        self.send(404,'text/plain',b'Image not found')
                elif path == '/accuracy.csv':
                    self.send(200, 'text/csv; charset=utf-8', (run/'accuracy.csv').read_bytes())
                elif path == '/healthz':
                    self.send(200, 'application/json', b'{"ok":true}')
                elif path == '/favicon.ico':
                    self.send(204, 'image/x-icon', b'')
                else: self.send(404, 'text/plain', b'Not found')
            except (OSError, ValueError, KeyError) as exc:
                self.log_error('%s', type(exc).__name__)
                self.send(503,'application/json',b'{"error":"Training data temporarily unavailable"}')
        do_HEAD = do_GET
        def log_message(self, fmt, *args):
            if len(args)>1 and str(args[1]) in {'200','204'}: return
            super().log_message(fmt,*args)
    return Handler

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18631)
    args=parser.parse_args()
    run=args.run_dir.resolve(strict=True)
    metrics(run)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(run))
    print('Serving read-only accuracy dashboard on port',args.port,flush=True)
    server.serve_forever()
