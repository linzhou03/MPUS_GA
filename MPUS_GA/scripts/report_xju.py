import json
import subprocess
import time
import sys

remote_code = """
import json
import glob
from collections import defaultdict

run_name = "cbst_quota_xju_fixed_20260922_s43_v1"
files = sorted(glob.glob("/home/gzw/projects/MPUS_GA/results_" + run_name + "/*/*.offline.json"))
by_dir = defaultdict(list)

for f in files:
    d = json.load(open(f))
    p = d.get('primary', {})
    rep = d.get('reporting', {})
    cm = p.get('confusion_matrix', [])
    dir_name = f.split('/')[-2]
    sub = f.split('/')[-1].replace('.offline.json', '')
    acc = p.get('accuracy', 0) * 100
    bal = p.get('balanced_accuracy', 0) * 100
    f1 = p.get('macro_f1', 0) * 100
    it = rep.get('selected_iteration')
    rec = p.get('per_class_recall', {})
    row_sums = [sum(row) for row in cm] if cm else []
    col_sums = [sum(cm[r][c] for r in range(len(cm))) for c in range(len(cm[0]))] if cm else []
    
    by_dir[dir_name].append({
        'sub': sub,
        'iter': it,
        'acc': acc,
        'bal': bal,
        'f1': f1,
        'pos_r': rec.get('positive', 0) * 100,
        'neu_r': rec.get('neutral', 0) * 100,
        'neg_r': rec.get('negative', 0) * 100,
        'preds': col_sums,
        'true': row_sums,
    })

print("---JSON_START---")
print(json.dumps(by_dir))
print("---JSON_END---")
"""

data = {}
for attempt in range(4):
    try:
        res = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', 'xju', 'python3'], input=remote_code, capture_output=True, text=True)
        if res.returncode == 0 and '---JSON_START---' in res.stdout:
            part = res.stdout.split('---JSON_START---')[1].split('---JSON_END---')[0].strip()
            data = json.loads(part)
            break
        else:
            print("SSH attempt returned:", res.stderr.strip() or "Empty stdout")
    except Exception as e:
        print("SSH error:", e)
    time.sleep(2)

if not data:
    print("Failed to get data from XJU.")
    sys.exit(1)

total_subjects = sum(len(v) for v in data.values())
print(f"XJU 总共已完成 {total_subjects} 个被试\n")

for d, entries in sorted(data.items()):
    mean_acc = sum(e['acc'] for e in entries) / len(entries)
    mean_bal = sum(e['bal'] for e in entries) / len(entries)
    mean_f1 = sum(e['f1'] for e in entries) / len(entries)
    mean_pos = sum(e['pos_r'] for e in entries) / len(entries)
    mean_neu = sum(e['neu_r'] for e in entries) / len(entries)
    mean_neg = sum(e['neg_r'] for e in entries) / len(entries)
    
    print(f"=================================================================")
    print(f"方向 {d} (完成 {len(entries)} 个被试)")
    print(f"=================================================================")
    print(f"  平均指标: Acc: {mean_acc:.2f}% | BalAcc: {mean_bal:.2f}% | Macro-F1: {mean_f1:.2f}%")
    print(f"  类别召回: 正向: {mean_pos:.1f}% | 中性: {mean_neu:.1f}% | 负向: {mean_neg:.1f}%\n")
    print(f"  各被试明细:")
    print(f"  {'被试':<20} | {'Acc':<7} | {'BalAcc':<7} | {'Macro-F1':<8} | {'中性Recall':<10} | {'正向Recall':<10} | {'负向Recall':<10} | {'预测分布 [正,中,负]'}")
    print(f"  {'-'*105}")
    for e in entries:
        print(f"  {e['sub']:<20} | {e['acc']:5.2f}% | {e['bal']:5.2f}% | {e['f1']:6.2f}% | {e['neu_r']:8.1f}% | {e['pos_r']:8.1f}% | {e['neg_r']:8.1f}% | {e['preds']}")
    print("\n")
