import glob
import json
import numpy as np

def get_cm(direction):
    files = sorted(glob.glob(f"/home/gzw/projects/MPUS_GA/results_cbst_quota_csu_best_20260922_s43_v1/{direction}/seed_*.json"))
    files = [f for f in files if not f.endswith('.offline.json')]
    total_cm = np.zeros((3, 3))
    for f in files:
        data = json.load(open(f))
        trace = data.get('target_evaluation_trace', [])
        post = [x for x in trace if x.get('iteration', 0) > 300]
        best_p = max(post, key=lambda x: x['evaluation']['fused']['balanced_accuracy'])
        cm = np.array(best_p['evaluation']['fused']['confusion_matrix'])
        norm_cm = cm / cm.sum(axis=1, keepdims=True)
        total_cm += norm_cm
    avg_cm = total_cm / len(files)
    return avg_cm

for d in ['A', 'B', 'C']:
    print(f"=== Direction {d} ===")
    cm = get_cm(d)
    for r in range(3):
        print(f"  {cm[r,0]:.4f}  {cm[r,1]:.4f}  {cm[r,2]:.4f}")
