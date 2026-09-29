# 混淆矩阵改进实验：ACE 三方向 LDS 数据

## 实验目的

针对 CAGA-SGA 论文 Figure 4 观察到的系统性混淆偏差（负向→正向 23-37%，中性→正向 24-40%），设计三类改进模块并验证其消融及组合效果。

## 核心问题

**CAGA-SGA 的"Positive 吸引子"现象**：
- 模型倾向于将 uncertain 样本预测为 positive
- 负向和中性大量混入 positive，但 positive 很少误判为其他类
- 不同迁移方向的混淆模式不对称

**本实验的三个改进方向**：
1. **非对称混淆损失（asy）**：对容易误判为 positive 的样本施加额外 margin
2. **增强偏置校准（cal）**：利用多尺度证据估计并校正类别预测偏置
3. **Confusion-Aware CBST（cbst）**：在伪标签选择时考虑混淆风险，对 positive 候选提高一致性要求

## 输入数据

使用固定参数 LDS 平滑后的数据：
- **路径**：`/home/gzw/projects/MPUS_GA/data_processed_ivfront_lds_a1c1q2r1p01_20260929_v1`
- **说明**：
  - SEED-IV 使用新的 trial 级前置滤波（0.1-70 Hz + 50 Hz 陷波）
  - 三数据集的 1/2/4 秒全部施加 LDS（A=1, C=1, Q=2, R=1, P0=1）
  - 前向 Kalman 滤波 + RTS 后向平滑
  - 已完成 519 个 NPZ 文件的结构和元数据校验

## 实验矩阵

```
方向: A (SEED-VII→SEED-V, 16 folds)
      C (SEED-IV→SEED-V, 16 folds)  
      E (SEED-IV→SEED-VII, 20 folds)

配置: baseline, asy, cal, cbst, asy_cal, asy_cbst, cal_cbst, full

总任务数: 3 方向 × 8 配置 = 24 个任务
```

### 配置说明

| 配置 ID | 模块组合 | 目的 |
|---------|----------|------|
| baseline | R4 原版 | 对照基线 |
| asy | +非对称混淆损失 | 单独验证损失改进 |
| cal | +增强偏置校准 | 单独验证校准改进 |
| cbst | +CA-CBST | 单独验证伪标签选择改进 |
| asy_cal | asy + cal | 验证损失与校准协同 |
| asy_cbst | asy + cbst | 验证损失与伪标签协同 |
| cal_cbst | cal + cbst | 验证校准与伪标签协同 |
| full | 全部三个模块 | 完整方案 |

## 训练协议

- **基础模型**：R4 原版（提交 5d239a754ee83baa435d660f96a9228880055599）
- **标准化**：`domain`（源域和目标域分别标准化）
- **CBST 变体**：`positive_gate`
- **随机种子**：43
- **训练步数**：1000
- **选优协议**：`post300_bal_best`（第 301-1000 步目标测试 BalAcc 最高检查点）
- **注**：选优使用目标真实标签，为诊断口径，非无标签 UDA

## 调度方式

**总队列 + 双卡并行接力**：
- 所有 24 个任务放入统一队列
- GPU 0 和 GPU 1 各自循环领取任务
- 任务完成后自动领取下一个，直到队列清空
- 避免固定分配导致的负载不均衡

**队列顺序**（按 Phase 分组）：
```
Phase 1: A_baseline, C_baseline, E_baseline
Phase 2: A_asy, C_asy, E_asy, A_cal, C_cal, E_cal, A_cbst, C_cbst, E_cbst
Phase 3: A_asy_cal, C_asy_cal, E_asy_cal, A_asy_cbst, C_asy_cbst, E_asy_cbst,
         A_cal_cbst, C_cal_cbst, E_cal_cbst, A_full, C_full, E_full
```

## 目录结构

```
/home/gzw/projects/MPUS_GA/
├── results_confusion_fix_lds_ace_20261001_v1/
│   ├── baseline/
│   │   ├── A/seed_43_subject_01.json ...
│   │   ├── C/seed_43_subject_01.json ...
│   │   └── E/seed_43_subject_01.json ...
│   ├── asy/
│   ├── cal/
│   ├── cbst/
│   ├── asy_cal/
│   ├── asy_cbst/
│   ├── cal_cbst/
│   └── full/
├── logs/confusion_fix_lds_ace_20261001_v1/
│   ├── suite_master.log
│   ├── suite_gpu0.log
│   ├── suite_gpu1.log
│   ├── baseline/
│   │   ├── A.log
│   │   ├── C.log
│   │   └── E.log
│   └── ...
```

## 关键指标

### 混淆矩阵相关
- **Neg→Pos**：负向误判为正向的比例（重点优化）
- **Neu→Pos**：中性误判为正向的比例（重点优化）
- **Pos precision**：正向预测的准确率
- **混淆矩阵对称性**：三类相互混淆的均衡程度

### 分类性能
- **Balanced Accuracy**：三类召回的平均（选优指标）
- **Pacc**：整体准确率
- **Macro F1**：宏平均 F1
- **Per-class Recall**：正向、中性、负向各自召回率

## 预期效果

### 短期目标（单模块）
- **asy**：Neg→Pos 和 Neu→Pos 各减少 5-10 个百分点
- **cal**：三类召回方差降低，BalAcc 提升 1-2 个百分点
- **cbst**：Positive precision 提升，减少 false positive

### 中期目标（组合）
- **asy_cal**：混淆偏差与性能同时改善
- **full**：在 A 方向达到或超越 CAGA-SGA 的对称性

### 长期目标
- 为 C/E 方向找到有效配置
- 建立针对不同迁移方向的自适应策略
- 消除"Positive 吸引子"的系统性偏差

## 执行计划

### 时间线
- **2026-10-01**：代码准备，创建实验分支，快速验证
- **2026-10-02**：启动双卡队列，完成 Phase 1 (baseline)
- **2026-10-03-04**：Phase 2 (单模块消融)
- **2026-10-05-06**：Phase 3 (组合实验)
- **2026-10-07**：结果汇总，生成对比报告

### 检查点
- 每完成一个 Phase，生成阶段性混淆矩阵对比
- 如果某模块效果不明显，可调整后续组合实验
- 如果发现明显改进，考虑加入 seed 42/44 验证

## 与前序实验的对比

| 项目 | 原版 R4 | IV 前置滤波 + LDS | 本实验（confusion fix） |
|------|---------|-------------------|------------------------|
| 数据 | data_processed | data_processed_ivfront_lds | 同左 |
| 方向 | A/B/C/D/E/F | C/D/E/F | A/C/E |
| 种子 | 43 | 43 | 43 |
| 配置 | 单一 | 单一 | 8 个消融 |
| 重点 | 基线性能 | LDS 平滑效果 | 混淆矩阵改进 |

## 技术细节

### 1. 非对称混淆损失（asy）
- 对"三尺度对 positive 意见不一致但融合预测 positive"的样本加大惩罚
- 对伪标签为 neg/neu 但靠近 positive 边界的样本施加 contrastive margin
- 权重：0.3

### 2. 增强偏置校准（cal）
- 每个尺度独立估计目标域 class prior（使用源域混淆矩阵 + 目标域软概率）
- 如果三尺度都估计出 positive 被高估，则强力校正
- 校正强度：0.5（温和版，避免 overcorrect）

### 3. Confusion-Aware CBST（cbst）
- 对 positive 类候选要求更高的三尺度一致性
- 计算每个候选的"混淆风险"（尺度分歧度）
- 选择风险最低的样本进入伪标签训练

## 代码位置

- **主训练脚本**：`trial_temporal/train_cbst.py`
- **非对称损失**：`trial_temporal/losses/asymmetric_confusion.py`
- **偏置校准**：`trial_temporal/calibration/multiscale_balance.py`
- **CA-CBST**：`trial_temporal/pseudo_label/confusion_aware_cbst.py`
- **调度器**：`scripts/master_worker_scheduler.py`
- **启动脚本**：`scripts/run_confusion_fix_suite.sh`

## 注意事项

1. **选模协议**：post300_bal_best 仍使用目标标签，为诊断口径
2. **数据路径**：确保 LDS 数据已完成 519 个 NPZ 的校验
3. **GPU 占用**：双卡并行，每卡独立调度
4. **中断恢复**：已完成的 fold 自动跳过，未完成的重新开始
5. **文件锁**：使用 fcntl 保证队列同步的线程安全

## 后续工作

根据本实验结果：
1. 如果 asy_cal 效果最好，考虑扩展到 B/D/F 方向
2. 如果单模块效果不明显，分析失败原因并改进
3. 如果混淆偏差显著改善，进行多种子验证（42/43/44）
4. 如果 C/E 方向仍不理想，设计方向特定的策略
5. 探索无目标标签的选模协议（如源域验证集、多尺度一致性）

## 参考

- **CAGA-SGA 论文**：Chen et al., "Golden-Style Graph Alignment for Cross-Dataset EEG Emotion Recognition", IEEE TAFC 2026
- **R4 基线**：提交 5d239a754ee83baa435d660f96a9228880055599
- **LDS 参数**：A=1, C=1, Q=2, R=1, P0=1（固定参数 Kalman 平滑）
- **前序实验**：`experiment/2026-09-29/固定参数LDS_XJU四方向`
