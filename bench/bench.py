#!/usr/bin/env python3
"""Arm runner: launch server via systemd-run, watchdog, smoke, decode panel, prefill ladder."""
import json, os, sys, time, subprocess, urllib.request, random, threading

ROOT = os.path.dirname(os.path.abspath(__file__))
URL = 'http://127.0.0.1:8002'
FLOOR_GIB = float(os.environ.get('FLOOR_GIB', '2.5'))
SWAP_GROWTH_GIB = float(os.environ.get('SWAP_GROWTH_GIB', '6'))


def meminfo():
    d = {}
    for l in open('/proc/meminfo'):
        k, v = l.split(':'); d[k] = int(v.split()[0]) * 1024
    return d


def gib(x): return round(x / 2**30, 2)


class Watch(threading.Thread):
    def __init__(s, unit):
        super().__init__(daemon=True); s.unit = unit; s.stop = False; s.breach = None
        m = meminfo(); s.swap0 = m['SwapTotal'] - m['SwapFree']; s.min_avail = m['MemAvailable']; s.max_swap = 0
    def run(s):
        while not s.stop:
            m = meminfo(); sw = m['SwapTotal'] - m['SwapFree'] - s.swap0
            s.min_avail = min(s.min_avail, m['MemAvailable']); s.max_swap = max(s.max_swap, sw)
            if m['MemAvailable'] < FLOOR_GIB * 2**30 or sw > SWAP_GROWTH_GIB * 2**30:
                s.breach = {'avail': gib(m['MemAvailable']), 'swap_growth': gib(sw), 't': time.time()}
                subprocess.run(['systemctl', '--user', 'kill', '-s', 'KILL', s.unit]); return
            time.sleep(0.5)


def req(path, body=None, timeout=1800):
    r = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body else None,
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def stream(messages, max_tokens, temp=0.0, extra=None):
    body = {'model': 'ornith-1.5-122b', 'messages': messages, 'max_tokens': max_tokens, 'stream': True,
            'temperature': temp, 'stream_options': {'include_usage': True}, 'timings_per_token': False}
    if temp == 0.0: body.update(top_k=1, seed=1)
    else: body.update(top_p=0.95, top_k=20, seed=1)
    if extra: body.update(extra)
    r = urllib.request.Request(URL + '/v1/chat/completions', data=json.dumps(body).encode(),
                               headers={'Content-Type': 'application/json'})
    t0 = time.time(); times = []; text = ''; reasoning = ''; usage = None; timings = None; finish = None
    with urllib.request.urlopen(r, timeout=3600) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(b'data: ') or line == b'data: [DONE]': continue
            ev = json.loads(line[6:])
            if ev.get('usage'): usage = ev['usage']
            if ev.get('timings'): timings = ev['timings']
            for ch in ev.get('choices', []):
                d = ch.get('delta', {})
                c = (d.get('content') or '') + (d.get('reasoning_content') or '')
                if c: times.append(time.time())
                text += d.get('content') or ''; reasoning += d.get('reasoning_content') or ''
                finish = ch.get('finish_reason') or finish
    ttft = times[0] - t0 if times else None
    return dict(ttft=ttft, wall=time.time() - t0, events=len(times), text=text, reasoning_len=len(reasoning),
                usage=usage, timings=timings, finish=finish)


PROMPTS = {
    'code': 'Write a complete Python implementation of a thread-safe LRU cache class with get, put, delete, resize and stats methods, with docstrings and type hints. Then write pytest tests for it.',
    'prose': 'Write a detailed long essay about the history of the printing press and its effects on European society, science and religion.',
    'math': 'Solve step by step: find all real x such that x^4 - 10x^3 + 35x^2 - 50x + 24 = 0, then compute the sum of squares of the roots, and prove the result two different ways.',
}


def filler(n_words, seed):
    rnd = random.Random(seed)
    words = ('system memory kernel latency throughput cache tensor router expert matrix vector token stream decode '
             'prefill scheduler batch queue graph node edge weight bias layer norm gradient optimizer checkpoint '
             'shard bucket hash index journal ledger audit policy service socket buffer packet frame').split()
    out = []
    for i in range(n_words):
        out.append(rnd.choice(words))
        if i % 17 == 16: out.append('.')
    return ' '.join(out)


def decode_panel(res, no_think=True):
    extra = {'chat_template_kwargs': {'enable_thinking': False}} if no_think else None
    for name, p in PROMPTS.items():
        r = stream([{'role': 'user', 'content': p}], 512, 0.0, extra)
        res['decode'][name] = summarize(r)
        print(name, res['decode'][name], flush=True)
    doc = filler(6000, 7)
    r = stream([{'role': 'user', 'content': 'Summarize this log in detail, section by section:\n' + doc}], 512, 0.0, extra)
    res['decode']['longdoc'] = summarize(r); print('longdoc', res['decode']['longdoc'], flush=True)
    r = stream([{'role': 'user', 'content': PROMPTS['code']}], 512, 0.6, extra)
    res['decode']['code_sampled'] = summarize(r); print('code_sampled', res['decode']['code_sampled'], flush=True)


def summarize(r):
    t = r['timings'] or {}
    return {k: v for k, v in dict(
        ttft=round(r['ttft'], 3) if r['ttft'] else None, prompt_n=t.get('prompt_n'), prompt_ms=t.get('prompt_ms'),
        prefill_tps=round(t.get('prompt_per_second', 0), 1), predicted_n=t.get('predicted_n'),
        decode_tps=round(t.get('predicted_per_second', 0), 2), draft_n=t.get('draft_n'),
        draft_acc=t.get('draft_n_accepted'), finish=r['finish'], events=r['events']).items()}


def prefill_ladder(res, sizes):
    for n in sizes:
        words = int(n * 0.92)
        nonce = f'NONCE-{time.time_ns()}-{random.random()}'
        content = nonce + '\n' + filler(words, n) + '\nReply with the single word OK.'
        r = stream([{'role': 'user', 'content': content}], 8, 0.0, {'chat_template_kwargs': {'enable_thinking': False}})
        s = summarize(r); res['prefill'][str(n)] = s; print('prefill', n, s, flush=True)


def main():
    arm = sys.argv[1]; env = json.loads(sys.argv[2]); sizes = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [2048, 8192, 32768, 131072]
    out = os.path.join(ROOT, 'runs', arm + '.json')
    if os.path.exists(out) and json.load(open(out)).get('status') == 'done':
        print('skip', arm); return
    unit = 'ornith122-serve'
    subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True)
    subprocess.run(['systemctl', '--user', 'reset-failed', unit], capture_output=True)
    time.sleep(3)
    w = Watch(unit); w.start()
    cmd = ['systemd-run', '--user', '--unit', unit, '--collect', '-p', 'MemorySwapMax=4G']
    for k, v in env.items(): cmd += ['--setenv', f'{k}={v}']
    cmd += ['--setenv', 'LLAMA_LOG_FILE=', '-p', f'StandardOutput=file:{ROOT}/runs/{arm}.log', '-p', f'StandardError=file:{ROOT}/runs/{arm}.log', 'bash', os.path.join(os.path.dirname(ROOT), 'serve.sh')]
    subprocess.run(cmd, check=True)
    res = {'arm': arm, 'env': env, 'status': 'loading', 'decode': {}, 'prefill': {}}
    t0 = time.time()
    try:
        while True:
            if w.breach: raise RuntimeError('watchdog breach during load %s' % w.breach)
            a = subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True).stdout.strip()
            if a not in ('active', 'activating'): raise RuntimeError('server died during load: ' + a)
            try:
                if req('/health', timeout=5).get('status') == 'ok': break
            except Exception: pass
            if time.time() - t0 > 1200: raise RuntimeError('load timeout')
            time.sleep(3)
        res['load_s'] = round(time.time() - t0, 1)
        m = meminfo(); res['avail_after_load_gib'] = gib(m['MemAvailable'])
        sm = stream([{'role': 'user', 'content': 'What is 17*23? Answer with just the number.'}], 2048, 0.0)
        res['smoke'] = {'text': sm['text'][-200:], 'reasoning_len': sm['reasoning_len'], 'finish': sm['finish'], 'ok': '391' in sm['text']}
        print('smoke', res['smoke'], flush=True)
        decode_panel(res)
        prefill_ladder(res, sizes)
        try:
            res['metrics'] = [l for l in urllib.request.urlopen(URL + '/metrics', timeout=10).read().decode().splitlines() if l.startswith('llamacpp:') and ('draft' in l or 'tokens' in l)]
        except Exception as e: res['metrics'] = str(e)
        res['status'] = 'done'
    except Exception as e:
        res['status'] = 'failed'; res['error'] = repr(e)
    finally:
        res['breach'] = w.breach; res['min_avail_gib'] = gib(w.min_avail); res['max_swap_growth_gib'] = gib(w.max_swap)
        w.stop = True
        if os.environ.get('KEEP') != '1':
            subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True)
        json.dump(res, open(out, 'w'), indent=1)
        print('RESULT', arm, res['status'], res.get('error'), 'min_avail', res['min_avail_gib'], 'swap+', res['max_swap_growth_gib'], flush=True)


if __name__ == '__main__':
    main()
