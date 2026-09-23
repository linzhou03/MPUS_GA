import json
import glob
import subprocess
import sys

script = """
import json, glob

def summarize(host, run):
    print(f"=== {host}: {run} ===")
    files = sorted(glob.glob(f"/home/gzw/projects/MPUS_GA/results_{run}/*/*.offline.json"))
    for f in files:
        d = json.load(open(f))
        p = d.get('primary', {})
        rep = d.get('reporting', {})
        cm = p.get('confusion_matrix', [])
        row_sums = [sum(row) for row in cm]
        col_sums = [sum(cm[r][c] for r in range(len(cm))) for c in range(len(cm[0]))] if cm else []
        dir_name = f.split('/')[-2]
        sub = f.split('/')[-1].replace('.offline.json', '')
        acc = p.get('accuracy', 0) * 100
        bal = p.get('balanced_accuracy', 0) * 100
        f1 = p.get('macro_f1', 0) * 100
        it = rep.get('selected_iteration')
        rec = p.get('per_class_recall', {})
        pos_r = rec.get('positive', 0) * 100
        neu_r = rec.get('neutral', 0) * 100
        neg_r = rec.get('negative', 0) * 100
        print(f"{dir_name} {sub} | Iter {it} | ACC {acc:.2f}% | BalAcc {bal:.2f}% | F1 {f1:.2f}%")
        print(f"   Recall -> Pos: {pos_r:.1f}%, Neu: {neu_r:.1f}%, Neg: {neg_r:.1f}%")
        print(f"   Preds (Pos,Neu,Neg) -> {col_sums} | True -> {row_sums}")
        print(f"   CM -> {cm}")

summarize("__HOST__", "__RUN__")
"""

for host, run in [('xju', 'cbst_quota_xju_fixed_20260922_s43_v1'), ('csu', 'cbst_quota_csu_best_20260922_s43_v1')]:
    h_script = script.replace('__HOST__', host).replace('__RUN__', run)
    res = subprocess.run(['ssh', '-o', 'ConnectTimeout=10', host, 'python3'], input=h_script, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
