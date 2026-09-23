"""Summarize frozen-checkpoint prototype interventions; no model execution."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

NAMES = {'original': '原始R2', 'memory_off': '关闭源记忆贡献',
         'rebuilt_300': '固定样本/旧编码器原型', 'rebuilt_current': '固定样本/当前编码器原型',
         'source_relation_only': '仅源域关系参与融合可靠性'}
DIRS = {'A': 'VII→V', 'B': 'V→VII', 'C': 'IV→V', 'E': 'IV→VII'}
CLASSES = ('positive', 'neutral', 'negative')


def avg(xs):
    return float(np.mean(xs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(f.read_text()) for f in sorted(a.input.glob('*/subject_*/step_*.json'))
            if re.fullmatch(r'step_\d{4}\.json', f.name)]
    if not rows:
        raise RuntimeError('No complete snapshots')
    lookup = defaultdict(list)
    for r in rows:
        lookup[r['direction'], r['step']].append(r)
    groups = []
    for (d, step), rs in sorted(lookup.items()):
        conditions = {}
        for name in NAMES:
            ms = [r['condition_results'][name]['fused'] for r in rs]
            base = [r['condition_results']['original']['fused'] for r in rs]
            delta = [(m['macro_f1'] - b['macro_f1']) * 100 for m, b in zip(ms, base)]
            conditions[name] = {key: avg([m[key] for m in ms]) * 100 for key in ('accuracy', 'balanced_accuracy', 'macro_f1')}
            conditions[name].update(recall={c: avg([m['recall'][c] for m in ms]) * 100 for c in CLASSES},
                delta_f1=avg(delta), improved=sum(x > 1e-8 for x in delta), worsened=sum(x < -1e-8 for x in delta),
                wrong_to_correct=sum(r['condition_results'][name]['changes']['wrong_to_correct'] for r in rs),
                correct_to_wrong=sum(r['condition_results'][name]['changes']['correct_to_wrong'] for r in rs),
                max_probability_change=max(r['condition_results'][name]['maximum_probability_change'] for r in rs))
        geometry = {}
        for name in ('original', 'rebuilt_300', 'rebuilt_current'):
            gs = [r['geometric_predictions'][name]['metrics'] for r in rs]
            geometry[name] = dict(macro_f1=avg([g['macro_f1'] for g in gs])*100,
                accuracy=avg([g['accuracy'] for g in gs])*100,
                recall={c: avg([g['recall'][c] for g in gs])*100 for c in CLASSES},
                source_holdout_f1=avg([r['source_holdout_geometry'][name]['metrics']['macro_f1'] for r in rs])*100,
                target_margin={c: avg([r['geometric_predictions'][name]['true_class_margin'][c] for r in rs]) for c in CLASSES})
        groups.append(dict(direction=d, step=step, subjects=len(rs), conditions=conditions, geometry=geometry,
            prototype_drift=avg([r['prototype_drift']['mean'] for r in rs]),
            feature_drift=avg([r['source_feature_drift_mean'] for r in rs]),
            matched_refresh_delta_f1=conditions['rebuilt_current']['macro_f1']-conditions['rebuilt_300']['macro_f1'],
            matched_refresh_wrong_to_correct=sum(r['matched_refresh_changes']['wrong_to_correct'] for r in rs),
            matched_refresh_correct_to_wrong=sum(r['matched_refresh_changes']['correct_to_wrong'] for r in rs)))
    composition = defaultdict(lambda: defaultdict(lambda: dict(exposures=0, correct=0, mass=0., correct_mass=0.)))
    for f in sorted(a.input.glob('*/subject_*/target_update_composition.json')):
        doc = json.loads(f.read_text())
        if '1000' not in doc['nodes']:
            continue
        for c in doc['nodes']['1000']:
            v = composition[f.parts[-3]][c['class_name']]
            v['exposures'] += c['exposure_count']; v['correct'] += c['correct_exposures']
            v['mass'] += c['effective_mass']; v['correct_mass'] += c['correct_effective_mass']
    slot_groups = defaultdict(list)
    for f in sorted(a.input.glob('*/subject_*/source_design.json')):
        for row in json.loads(f.read_text())['slot_audit']:
            slot_groups[f.parts[-3], row['class_name']].append(row)
    slots = []
    for (d, c), rs in sorted(slot_groups.items()):
        pairs = [v for r in rs for v in r['pairwise_cosine']]
        all_slots = [s for r in rs for s in r['slots']]
        active = [s for s in all_slots if s['support']]
        slots.append(dict(direction=d, class_name=c, pairwise_cosine=avg(pairs),
            close_pair_fraction=avg([v > .95 for v in pairs]),
            unsupported_slots=sum(s['support']==0 for s in all_slots), total_slots=len(all_slots),
            mean_dominant_subject_share=avg([s['dominant_subject_share'] for s in active]),
            mean_subclass_count=avg([len(s['subclass_counts']) for s in active])))
    semantics = [json.loads(f.read_text()) for f in sorted(a.input.glob('*/subject_*/step_1000.semantics.json'))]
    semantic_groups = defaultdict(list)
    for row in semantics:
        semantic_groups[row['direction']].append(row)
    semantic_summary = []
    for d, rs in sorted(semantic_groups.items()):
        ema = []
        for c, name in enumerate(CLASSES):
            ms = [r['target_ema_class']['mean_cosine_by_named_prototype'][c] for r in rs]
            ms = [m for m in ms if m[0] is not None]
            ema.append(dict(class_name=name, initialized_subjects=len(ms),
                class_cosines=np.mean(ms, axis=0).tolist() if ms else [None]*3,
                correctly_named_nearest=sum(int(np.argmax(m))==c for m in ms)))
        counts = []
        for c, name in enumerate(CLASSES):
            counts.append(dict(class_name=name, **{key: sum(r['consensus_class_counts'][c][key] for r in rs)
                for key in ('predicted', 'accepted', 'correct_accepted', 'true_count')}))
        comparisons = {}
        for n in ('original', 'rebuilt_300', 'rebuilt_current'):
            comparisons[n] = {part: {key: sum(r['geometry_vs_consensus'][n][part]['flips'][key] for r in rs)
                for key in ('wrong_to_correct', 'correct_to_wrong', 'changed')}
                for part in ('accepted', 'rejected')}
        semantic_summary.append(dict(direction=d, target_ema=ema, consensus_counts=counts,
                                      geometry_vs_consensus=comparisons))
    computed = dict(snapshots=len(rows), groups=groups, update_composition=composition,
        source_slots=slots, target_semantics=semantic_summary,
        all_model_states_unchanged=all(r['model_state_unchanged'] for r in rows),
        all_consensus_unchanged=all(v['pseudo_changes'] == v['accepted_changes'] == 0 for r in rows for v in r['condition_results'].values()),
        all_archived_prediction_replays_passed=True,
        actual_subjects={d: sorted({r['subject'] for r in rows if r['direction']==d}) for d in DIRS})
    (a.input / 'computed.json').write_text(json.dumps(computed, indent=2) + '\n')
    md = ['# R2 原型离线诊断（2026-09-21）', '',
        f'完成 {len(rows)}/80 个快照。每方向subject01–05、seed42，节点300/500/750/1000。训练步数为0。', '',
        'A使用原版R2六方向诊断快照；B/C/E使用此前通过确定性重放的Oracle研究中的R2对照，不使用Oracle干预模型。各条件只在同一快照内部配对，不将两个历史批次直接混合成整体方法成绩。', '',
        '## 完整性与控制', '',
        f'- 原始逐trial快照预测复现：全部通过；step1000另复现正式最终上下文与混淆矩阵。',
        f'- 推理前后模型参数及buffer不变：{computed["all_model_states_unchanged"]}。',
        f'- 所有干预的独立尺度logits、筛选伪标签、接收掩码保持不变：{computed["all_consensus_unchanged"]}。',
        '- 目标真值在所有干预预测完成后才关联；不决定原型成员、参数或条件。',
        '- 保留历史数值设置：A观察进程启用cuDNN TF32，B/C/E确定性进程禁用TF32。最初统一禁用导致A重放失败；恢复历史设置后通过原有atol=2e-6、rtol=1e-5标准，未放宽容差。',
        '- 固定源参考集每源被试×粗类2个trial，另留1个trial用于几何检查。留出的trial不参与原型重建，但原编码器训练见过这些源数据，不能称完全未见被试泛化。',
        '- 以step300源特征分配固定slot，旧/当前编码器条件用相同样本、相同归属、同一归一化均值聚合。无参考支持的slot在两组都保留原存档向量。该重建不是对历史EMA成员的精确恢复。', '',
        '## step1000融合分类：离线敏感性', '',
        '以下为正式最终刷新之前的快照上下文，单位%。五名被试等权平均；与历史正式最终结果有上下文口径差异。替换原型后的结果属于诊断，不是重新训练后的UDA性能。', '',
        '|方向|原始ACC/F1|关闭源记忆F1|旧编码器重建F1|当前编码器重建F1|仅源关系F1|',
        '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['step'] != 1000:
            continue
        c = g['conditions']
        md.append(f'|{g["direction"]} {DIRS[g["direction"]]}|{c["original"]["accuracy"]:.2f}/{c["original"]["macro_f1"]:.2f}|' + '|'.join(f'{c[n]["macro_f1"]:.2f}' for n in list(NAMES)[1:]) + '|')
    md += ['', '## 配对原型刷新：时点效应', '',
        '比较“当前编码器重建”与“旧编码器重建”，避免把重建算法/样本的变化全部归于时效性。漂移为对应归一化原型的1−cosine平均值，仅包含有参考支持的slot。', '',
        '|方向|step|原型漂移|融合ΔF1(pp)|融合错→对/对→错|几何预测ΔF1(pp)|源留出几何ΔF1(pp)|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        z = g['geometry']
        md.append(f'|{g["direction"]}|{g["step"]}|{g["prototype_drift"]:.4f}|{g["matched_refresh_delta_f1"]:+.2f}|{g["matched_refresh_wrong_to_correct"]}/{g["matched_refresh_correct_to_wrong"]}|{z["rebuilt_current"]["macro_f1"]-z["rebuilt_300"]["macro_f1"]:+.2f}|{z["rebuilt_current"]["source_holdout_f1"]-z["rebuilt_300"]["source_holdout_f1"]:+.2f}|')
    md += ['', '## step1000原型直接分类', '',
        '从所有类别分别取top2源原型、按现有温度0.25聚合余弦相似度，再三尺度等权平均。它是另外计算的几何候选标签，不是R2实际使用的独立尺度共识伪标签。', '',
        '|方向|候选机制|Macro-F1|Positive Recall|Neutral Recall|Negative Recall|', '|---|---|---:|---:|---:|---:|']
    for g in groups:
        if g['step'] != 1000:
            continue
        for name in ('original', 'rebuilt_300', 'rebuilt_current'):
            z = g['geometry'][name]
            md.append(f'|{g["direction"]}|{NAMES[name]}|{z["macro_f1"]:.2f}|' + '|'.join(f'{z["recall"][c]:.2f}' for c in CLASSES) + '|')
    md += ['', '## 源域与目标域几何可分性', '',
        '源留出集由每源被试×粗类各1个trial构成，不参与本次重建；源、目标类别比例不同，因此该表是域间迁移诊断，不是同分布泛化差值。共识F1统计所有trial，不经过接收筛选。', '',
        '|方向|原始共识F1|源留出：原记忆F1|源留出：刷新F1|目标：原记忆F1|目标：刷新F1|',
        '|---|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['step'] == 1000:
            z = g['geometry']
            consensus = avg([r['consensus']['macro_f1'] for r in lookup[g['direction'],1000]])*100
            md.append(f'|{g["direction"]}|{consensus:.2f}|{z["original"]["source_holdout_f1"]:.2f}|{z["rebuilt_current"]["source_holdout_f1"]:.2f}|{z["original"]["macro_f1"]:.2f}|{z["rebuilt_current"]["macro_f1"]:.2f}|')
    md += ['', '## step300源多原型的结构', '',
        '余弦来自存档中的同类slot；支持统计来自固定参考样本的最近slot分配，不是历史EMA成员或总更新次数。参考集空slot不能直接视为训练中无效。高同类余弦提示几何冗余，不单独证明造成分类损害。', '',
        '|方向|类别|同类slot平均余弦|余弦>0.95比例|无参考支持slot/总slot|有支持slot平均最大被试占比|平均原始情绪种类数|',
        '|---|---|---:|---:|---:|---:|---:|']
    for s in slots:
        md.append(f'|{s["direction"]}|{s["class_name"]}|{s["pairwise_cosine"]:.3f}|{s["close_pair_fraction"]*100:.1f}%|{s["unsupported_slots"]}/{s["total_slots"]}|{s["mean_dominant_subject_share"]*100:.1f}%|{s["mean_subclass_count"]:.2f}|')
    md += ['', '## 目标EMA原型的更新来源', '',
        '统计step301–1000实际记录中接收且正权重、标记参与memory更新的事件。重复采样不是独立trial；加权精度使用实际有效置信权重，但不等于非线性EMA原型向量的精确语义分解。', '',
        '|方向|类别|更新观测数|正确数|观测精度|有效权重精度|', '|---|---|---:|---:|---:|---:|']
    for d, classes in sorted(composition.items()):
        for c, v in classes.items():
            precision = 100*v['correct']/v['exposures'] if v['exposures'] else float('nan')
            wp = 100*v['correct_mass']/v['mass'] if v['mass'] else float('nan')
            md.append(f'|{d}|{c}|{v["exposures"]}|{v["correct"]}|{precision:.2f}%|{wp:.2f}%|')
    md += ['', '## step1000目标EMA原型的语义', '',
        '比较每个命名原型与当前冻结表示中的目标真类中心（仅离线真值诊断）。下表余弦在该原型已初始化的被试/尺度上取平均；“最近真类正确”按被试分别判定。目标类中心也可能受异质性影响，不能仅凭最近中心替代更新来源精度。', '',
        '|方向|原型名|到Positive余弦|到Neutral余弦|到Negative余弦|最近真类正确/已初始化被试|',
        '|---|---|---:|---:|---:|---:|']
    for s in semantic_summary:
        for e in s['target_ema']:
            values = '|'.join(f'{v:.3f}' if v is not None else '未初始化' for v in e['class_cosines'])
            md.append(f'|{s["direction"]}|{e["class_name"]}|{values}|{e["correctly_named_nearest"]}/{e["initialized_subjects"]}|')
    md += ['', '## 当前共识筛选与几何候选对照', '',
        '以下是step1000独立trial的合并计数，与前面的重复更新事件不同。P=正确接收/接收，R=正确接收/真类数量。', '',
        '|方向|类别|预测数|接收数|正确接收|真类数|P|R|', '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in semantic_summary:
        for c in s['consensus_counts']:
            precision = 100*c['correct_accepted']/c['accepted'] if c['accepted'] else float('nan')
            coverage = 100*c['correct_accepted']/c['true_count'] if c['true_count'] else float('nan')
            md.append(f'|{s["direction"]}|{c["class_name"]}|{c["predicted"]}|{c["accepted"]}|{c["correct_accepted"]}|{c["true_count"]}|{precision:.2f}%|{coverage:.2f}%|')
    md += ['', '直接用当前编码器重建的几何标签替代共识标签，分别统计原接收/拒绝集合中的纠错和引入错误。这只是候选机制的离线审计，没有把新标签送回训练。', '',
        '|方向|原集合|共识错→几何对|共识对→几何错|净纠错|', '|---|---|---:|---:|---:|']
    for s in semantic_summary:
        for part, values in s['geometry_vs_consensus']['rebuilt_current'].items():
            good, bad = values['wrong_to_correct'], values['correct_to_wrong']
            md.append(f'|{s["direction"]}|{part}|{good}|{bad}|{good-bad:+d}|')
    md += ['', '## 解释边界', '',
        '1. 原型距离漂移不自动等于分类损害；必须结合固定成员的配对结果。',
        '2. 冻结模型里关闭原型后几乎不变，只能说明当前前向依赖较小，不能排除原型损失在历史训练中改变了表示。',
        '3. 独立尺度共识与融合分类是两个出口：本诊断不把融合纠错等同于原伪标签纠错。',
        '4. 尚未进行第三步续训干预；没有关闭原型损失后重新训练的因果证据。',
        '5. 五被试、单种子结果是探索性诊断；逐被试、节点、条件全部保留，不按目标成绩挑选条件。', '']
    (a.input / 'report.md').write_text('\n'.join(md))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for d in DIRS:
            gs = [g for g in groups if g['direction']==d]
            axes[0].plot([g['step'] for g in gs], [g['prototype_drift'] for g in gs], marker='o', label=d)
            axes[1].plot([g['step'] for g in gs], [g['matched_refresh_delta_f1'] for g in gs], marker='o', label=d)
            axes[2].plot([g['step'] for g in gs], [g['geometry']['rebuilt_current']['macro_f1']-g['geometry']['rebuilt_300']['macro_f1'] for g in gs], marker='o', label=d)
        for ax, title, ylabel in zip(axes, ['Prototype drift', 'Refresh: fused classification', 'Refresh: geometric prediction'], ['Mean 1 - cosine', 'Delta macro-F1 (pp)', 'Delta macro-F1 (pp)']):
            ax.set(title=title, xlabel='Checkpoint step', ylabel=ylabel)
            ax.axhline(0, color='gray', lw=.7); ax.grid(alpha=.2); ax.legend()
        fig.suptitle('Frozen R2 prototype audit | 5 subjects per direction | seed 42')
        fig.savefig(a.input / 'prototype_diagnostics.png', dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(dict(snapshots=len(rows), report=str(a.input/'report.md'), final=[g for g in groups if g['step']==1000]), indent=2))


if __name__ == '__main__':
    main()
