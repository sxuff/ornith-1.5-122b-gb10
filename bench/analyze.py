import json, glob, os
ROOT = os.path.dirname(os.path.abspath(__file__))
for f in sorted(glob.glob(ROOT + '/runs/*.json')):
    if 'killed' in f: continue
    r = json.load(open(f))
    d = r.get('decode', {}); p = r.get('prefill', {})
    dec = {k: v.get('decode_tps') for k, v in d.items()}
    acc = {k: (round(v['draft_acc'] / v['draft_n'], 2) if v.get('draft_n') else None) for k, v in d.items()}
    vals = [v for k, v in dec.items() if v and k != 'code_sampled']
    mean = round(sum(vals) / len(vals), 2) if vals else None
    print(r['arm'], r['status'], 'mean4', mean, dec, 'acc', acc,
          'prefill', {k: v.get('prefill_tps') for k, v in p.items()},
          'minavail', r.get('min_avail_gib'), 'swap+', r.get('max_swap_growth_gib'), 'smoke', r.get('smoke', {}).get('ok'))
