import json
import subprocess
import time
import sys

def get_remote_data(host, run_name):
    # Use standard string format instead of f-string to avoid brace escaping errors
    script_template = """
import json
import glob
from collections import defaultdict

run_name = "__RUN_NAME__"
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
    
    entry = {
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
    }
    
    # Also check post-300 best BalAcc if raw json has target_evaluation_trace
    raw_json_path = f.replace('.offline.json', '.json')
    try:
        raw_d = json.load(open(raw_json_path))
        trace = raw_d.get('target_evaluation_trace', [])
        trace_post = [x for x in trace if x.get('iteration', 0) > 300]
        if trace_post:
            best_post = max(trace_post, key=lambda x: x['evaluation']['fused']['balanced_accuracy'])
            m_post = best_post['evaluation']['fused']
            r_post = m_post['per_class_recall']
            entry['post300_bal'] = {
                'iter': best_post['iteration'],
                'acc': m_post['accuracy'] * 100,
                'bal': m_post['balanced_accuracy'] * 100,
                'f1': m_post['macro_f1'] * 100,
                'pos_r': r_post.get('positive', 0) * 100,
                'neu_r': r_post.get('neutral', 0) * 100,
                'neg_r': r_post.get('negative', 0) * 100,
                'preds': m_post.get('prediction_counts', {})
            }
    except Exception:
        pass
        
    by_dir[dir_name].append(entry)

print("---JSON_START---")
print(json.dumps(by_dir))
print("---JSON_END---")
"""
    script = script_template.replace("__RUN_NAME__", run_name)
    for attempt in range(4):
        try:
            res = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', host, 'python3'], input=script, capture_output=True, text=True)
            if res.returncode == 0 and '---JSON_START---' in res.stdout:
                part = res.stdout.split('---JSON_START---')[1].split('---JSON_END---')[0].strip()
                return json.loads(part)
        except Exception:
            pass
        time.sleep(2)
    return {}

print("Fetching XJU results...")
xju_data = get_remote_data('xju', 'cbst_quota_xju_fixed_20260922_s43_v1')

print("Fetching CSU results...")
csu_data = get_remote_data('csu', 'cbst_quota_csu_best_20260922_s43_v1')

def print_summary(host_title, data, is_csu=False):
    print(f"\n{'='*75}")
    print(f"  {host_title}")
    print(f"{'='*75}")
    if not data:
        print("  No data returned.")
        return

    for direction, entries in sorted(data.items()):
        print(f"\n>>> 方向 {direction} (已完成 {len(entries)} 个被试) <<<")
        accs = [e['acc'] for e in entries]
        bals = [e['bal'] for e in entries]
        f1s = [e['f1'] for e in entries]
        pos_rs = [e['pos_r'] for e in entries]
        neu_rs = [e['neu_r'] for e in entries]
        neg_rs = [e['neg_r'] for e in entries]

        mean_acc = sum(accs) / len(accs)
        mean_bal = sum(bals) / len(bals)
        mean_f1 = sum(f1s) / len(f1s)
        mean_pos = sum(pos_rs) / len(pos_rs)
        mean_neu = sum(neu_rs) / len(neu_rs)
        mean_neg = sum(neg_rs) / len(neg_rs)

        print(f"  【已完成 {len(entries)} 个被试平均】")
        print(f"    总体 Acc: {mean_acc:.2f}% | 均衡准度 BalAcc: {mean_bal:.2f}% | 宏平均 Macro-F1: {mean_f1:.2f}%")
        print(f"    各类别召回率 -> 正向: {mean_pos:.1f}%, 中性: {mean_neu:.1f}%, 负向: {mean_neg:.1f}%")

        if is_csu and any('post300_bal' in e for e in entries):
            p_accs = [e['post300_bal']['acc'] for e in entries if 'post300_bal' in e]
            p_bals = [e['post300_bal']['bal'] for e in entries if 'post300_bal' in e]
            p_f1s = [e['post300_bal']['f1'] for e in entries if 'post300_bal' in e]
            p_pos = [e['post300_bal']['pos_r'] for e in entries if 'post300_bal' in e]
            p_neu = [e['post300_bal']['neu_r'] for e in entries if 'post300_bal' in e]
            p_neg = [e['post300_bal']['neg_r'] for e in entries if 'post300_bal' in e]
            print(f"  【CSU 300步后平衡选优(Post-300 BalBest)策略平均 ({len(p_accs)} 个被试)】")
            print(f"    总体 Acc: {sum(p_accs)/len(p_accs):.2f}% | 均衡准度 BalAcc: {sum(p_bals)/len(p_bals):.2f}% | 宏平均 Macro-F1: {sum(p_f1s)/len(p_f1s):.2f}%")
            print(f"    各类别召回率 -> 正向: {sum(p_pos)/len(p_pos):.1f}%, 中性: {sum(p_neu)/len(p_neu):.1f}%, 负向: {sum(p_neg)/len(p_neg):.1f}%")

        print("  【各被试明细】")
        for e in entries:
            it_str = str(e['iter']) if e['iter'] is not None else "None"
            print(f"    {e['sub']} | Iter: {it_str:4s} | Acc: {e['acc']:5.2f}% | BalAcc: {e['bal']:5.2f}% | F1: {e['f1']:5.2f}% | 中性Recall: {e['neu_r']:5.1f}% | 预测:[正,中,负]={e['preds']}")
            if is_csu and 'post300_bal' in e:
                pb = e['post300_bal']
                print(f"      ↳ [Post-300 BalBest] Iter: {pb['iter']:4d} | Acc: {pb['acc']:5.2f}% | BalAcc: {pb['bal']:5.2f}% | F1: {pb['f1']:5.2f}% | 中性Recall: {pb['neu_r']:5.1f}%")

print_summary("XJU 结果: Fixed Final (固定第1000步最终模型，真实无早停无监督 UDA)", xju_data, is_csu=False)
print_summary("CSU 结果: Test Best (对比 CAGA-SGA 原生 Acc 峰值 vs 适应期平衡选优)", csu_data, is_csu=True)
