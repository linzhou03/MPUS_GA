import json
import subprocess
import time
import sys

script = """
import json, glob
from collections import defaultdict

run_name = 'cbst_quota_csu_best_20260922_s43_v1'
files = sorted(glob.glob('/home/gzw/projects/MPUS_GA/results_' + run_name + '/*/*.offline.json'))

by_dir_raw = defaultdict(list)
by_dir_post = defaultdict(list)

for f in files:
    d = json.load(open(f))
    p = d.get('primary', {})
    rec = p.get('per_class_recall', {})
    dir_name = f.split('/')[-2]
    by_dir_raw[dir_name].append({
        'acc': p.get('accuracy', 0),
        'bal': p.get('balanced_accuracy', 0),
        'f1': p.get('macro_f1', 0),
        'pos': rec.get('positive', 0),
        'neu': rec.get('neutral', 0),
        'neg': rec.get('negative', 0)
    })
    raw_f = f.replace('.offline.json', '.json')
    try:
        raw_d = json.load(open(raw_f))
        trace = raw_d.get('target_evaluation_trace', [])
        post = [x for x in trace if x.get('iteration', 0) > 300]
        if post:
            best_p = max(post, key=lambda x: x['evaluation']['fused']['balanced_accuracy'])
            m_p = best_p['evaluation']['fused']
            r_p = m_p.get('per_class_recall', {})
            by_dir_post[dir_name].append({
                'acc': m_p.get('accuracy', 0),
                'bal': m_p.get('balanced_accuracy', 0),
                'f1': m_p.get('macro_f1', 0),
                'pos': r_p.get('positive', 0),
                'neu': r_p.get('neutral', 0),
                'neg': r_p.get('negative', 0)
            })
    except Exception:
        pass

out = {'raw': by_dir_raw, 'post': by_dir_post}
print('---JSON_START---')
print(json.dumps(out))
print('---JSON_END---')
"""

res = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', 'csu', 'python3'], input=script, capture_output=True, text=True)
if res.returncode == 0 and '---JSON_START---' in res.stdout:
    part = res.stdout.split('---JSON_START---')[1].split('---JSON_END---')[0].strip()
    data = json.loads(part)
    raw = data['raw']
    post = data['post']
    
    print("CSU 进度与指标对比:")
    print("=" * 75)
    for d in ['A', 'B', 'C', 'D', 'E', 'F']:
        r_items = raw.get(d, [])
        p_items = post.get(d, [])
        if not r_items:
            print(f"方向 {d}: 尚未开始或无完成折")
            continue
        print(f"方向 {d} (完成 {len(r_items)} 折):")
        
        # Raw CAGA-SGA style
        r_acc = sum(x['acc'] for x in r_items)/len(r_items)*100
        r_bal = sum(x['bal'] for x in r_items)/len(r_items)*100
        r_f1 = sum(x['f1'] for x in r_items)/len(r_items)*100
        r_pos = sum(x['pos'] for x in r_items)/len(r_items)*100
        r_neu = sum(x['neu'] for x in r_items)/len(r_items)*100
        r_neg = sum(x['neg'] for x in r_items)/len(r_items)*100
        print(f"  [原生 Acc 选优]  Acc: {r_acc:5.2f}% | BalAcc: {r_bal:5.2f}% | F1: {r_f1:5.2f}% | Pos: {r_pos:5.1f}% | Neu: {r_neu:5.1f}% | Neg: {r_neg:5.1f}%")
        
        if p_items:
            p_acc = sum(x['acc'] for x in p_items)/len(p_items)*100
            p_bal = sum(x['bal'] for x in p_items)/len(p_items)*100
            p_f1 = sum(x['f1'] for x in p_items)/len(p_items)*100
            p_pos = sum(x['pos'] for x in p_items)/len(p_items)*100
            p_neu = sum(x['neu'] for x in p_items)/len(p_items)*100
            p_neg = sum(x['neg'] for x in p_items)/len(p_items)*100
            print(f"  [300步后BalBest] Acc: {p_acc:5.2f}% | BalAcc: {p_bal:5.2f}% | F1: {p_f1:5.2f}% | Pos: {p_pos:5.1f}% | Neu: {p_neu:5.1f}% | Neg: {p_neg:5.1f}%")
        print("-" * 75)
else:
    print("Failed to get CSU data:", res.stderr)
