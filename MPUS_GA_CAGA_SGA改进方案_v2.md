# MPUS-GA：多原型可靠度驱动的风格—语义图协同对齐

> Multi-Prototype Uncertainty-aware Style and Graph Alignment for Cross-Dataset EEG Emotion Recognition
>
> 修订版 v2：面向 CAGA-SGA 的可实现方案、实验设计与论文表述

---

## 1. 核心结论

CAGA-SGA 的两个潜在瓶颈是：

1. 单一 Golden Style 只能描述一个全局统计中心，难以覆盖跨被试、跨设备和跨数据集形成的多模态风格分布；
2. SGA 将目标域高置信度伪标签直接转换为硬图边，稳定但错误的伪标签会产生系统性错误拓扑。

本方案提出 **MPUS-GA**，但不将“多原型”和“不确定性软图”作为两个相互独立的外挂模块，而是引入统一的目标样本可靠度：

\[
r_i \in [0,1]
\]

该可靠度同时控制：

- 目标样本对 Golden Style prototype 学习与路由对齐的贡献；
- 目标样本对 source-target、target-target 语义图边监督的贡献。

因此，完整逻辑为：

```text
目标域 EMA 教师预测
        │
        ├── 预测熵：样本是否含糊
        └── MC 分歧：模型是否不稳定
                    │
                    ▼
             样本可靠度 r_i
              ┌─────┴─────┐
              ▼           ▼
      可靠多原型风格学习   可靠软语义图监督
              └─────┬─────┘
                    ▼
              跨数据集情绪分类
```

本方案的首要目标不是增加模块数量，而是降低错误目标信息同时污染风格空间和语义拓扑的风险。

---

## 2. 任务设定与约束

给定有标签源域：

\[
\mathcal D_s=\{(x_i^s,y_i^s)\}_{i=1}^{N_s}
\]

以及无标签目标域：

\[
\mathcal D_t=\{x_j^t\}_{j=1}^{N_t},
\]

目标是在训练过程中不访问目标域标签，仅使用目标 EEG 特征完成无监督域适配，并在固定训练预算结束后进行一次目标域评估。

当前任务包含：

- SEED-VII \(\rightarrow\) SEED-V；
- SEED-IV \(\rightarrow\) SEED-V；
- 统一映射为 positive、neutral、negative 三分类。

由于三个数据集的类别先验不同，不能直接假设源域和目标域的总体 prototype 使用比例相同：

- SEED-VII：30% / 10% / 60%；
- SEED-IV：25% / 25% / 50%；
- SEED-V：20% / 20% / 60%。

因此，本文采用**类别条件路由对齐**，不使用无条件 batch 均值路由对齐。

---

## 3. 基线与问题定位

### 3.1 单一 Golden Style

原方法在第 \(l\) 层学习一个全局统计锚点：

\[
G_l=(\mu_{g,l},\sigma_{g,l}).
\]

对样本特征 \(Z_{i,l}\in\mathbb R^{T\times D}\) 执行：

\[
\hat Z_{i,l}
=\sigma_{g,l}\odot
\frac{Z_{i,l}-\mu(Z_{i,l})}{\sigma(Z_{i,l})+\epsilon}
+\mu_{g,l}.
\]

所有样本共享同一目标统计量，隐含假设是跨数据集风格偏移近似单峰分布。实际 EEG 特征可能由被试、采集协议、脑网络状态等多个潜在子域共同构成，单一统计中心可能产生过度对齐。

### 3.2 硬伪标签语义图

原 SGA 先得到目标预测：

\[
\hat y_j^t=\arg\max_c p_{jc}^t,
\]

再生成二值理想邻接矩阵：

\[
A_{ij}^{\ast}=\mathbb I(\hat y_i=\hat y_j).
\]

置信度阈值可以过滤一部分低质量目标节点，但无法处理“高置信度且稳定地预测错误”的样本。一个错误目标标签还会同时影响与其相连的多条边，使点级错误扩展为图级结构污染。

---

## 4. MPUS-GA 总体框架

MPUS-GA 包含四个关键部件：

1. EMA Teacher：产生比即时 student 更平滑的目标预测；
2. 双因素可靠度：同时刻画预测含糊度和模型分歧；
3. Reliability-aware Multi-Prototype Golden Alignment，简称 R-MPGA；
4. Reliability-aware Soft SGA，简称 R-SoftSGA。

训练流程如下：

```text
Source EEG ──┐
             ├─ Student Encoder ─ Multi-Prototype Style Alignment ─ Classifier
Target EEG ──┘                                │
                                             ├─ Domain alignment
                                             └─ Soft semantic graph

Target EEG ─ EMA Teacher + stochastic inference ─ q_i, r_i
                                                    │
                         ┌──────────────────────────┴────────────────────┐
                         ▼                                               ▼
            加权目标 prototype/route 损失                    加权 ST/TT 图边损失
```

Teacher 只由 student 参数的指数滑动平均更新，不参与梯度反向传播。

---

## 5. 可靠度估计

### 5.1 EMA Teacher

记 student 参数为 \(\theta\)，teacher 参数为 \(\bar\theta\)。每次 student 更新后：

\[
\bar\theta\leftarrow
\rho\bar\theta+(1-\rho)\theta,
\]

其中推荐初值：

\[
\rho=0.99.
\]

训练开始时直接复制 student 参数初始化 teacher。

### 5.2 MC Dropout 预测

对同一目标样本执行 \(M\) 次随机前向：

\[
p_i^{(m)}
=\operatorname{softmax}\left(z_i^{(m)}/T_{\mathrm{cal}}\right),
\qquad m=1,\dots,M.
\]

平均预测为：

\[
q_i=\frac{1}{M}\sum_{m=1}^{M}p_i^{(m)}.
\]

其中：

- \(M=3\) 用于快速开发；
- \(M=5\) 用于正式实验；
- \(T_{\mathrm{cal}}\) 只能由源域验证集校准，不能使用目标标签；若不做校准则固定为 1，并在论文中明确说明。

实现时 teacher 整体处于 evaluation 状态，只临时开启 Dropout 层，避免其他带运行统计量的层在随机推理时被更新。

### 5.3 预测含糊度

归一化预测熵为：

\[
u_i^{\mathrm{ent}}
=\frac{H(q_i)}{\log C},
\]

其中：

\[
H(q_i)=-\sum_{c=1}^{C}q_{ic}\log(q_{ic}+\epsilon).
\]

当预测接近均匀分布时，\(u_i^{\mathrm{ent}}\) 接近 1。

### 5.4 模型分歧

使用预测熵与期望熵之差刻画随机前向之间的分歧：

\[
u_i^{\mathrm{mi}}
=\frac{
H(q_i)-\frac1M\sum_{m=1}^{M}H(p_i^{(m)})
}{\log C}.
\]

数值实现时将其截断到非负区间：

\[
u_i^{\mathrm{mi}}\leftarrow\max(0,u_i^{\mathrm{mi}}).
\]

### 5.5 最终可靠度

定义：

\[
r_i
=\left(1-u_i^{\mathrm{ent}}\right)
\exp\left(-\beta u_i^{\mathrm{mi}}\right),
\]

并截断到 \([0,1]\)。推荐：

\[
\beta=2.
\]

该定义同时排除两类不可靠样本：

- 多次预测一致但接近均匀分布的样本；
- 平均预测看似明确、但不同随机前向严重分歧的样本。

相比 \(r_i=1-u_i\)，它不会把“稳定的均匀预测”误判为最高可靠度。

---

## 6. Reliability-aware Multi-Prototype Golden Alignment

### 6.1 多原型 Golden Style Bank

将第 \(l\) 层的单一 Golden Style 扩展为：

\[
\mathcal G_l
=\{G_{l,1},G_{l,2},\dots,G_{l,K}\},
\]

其中：

\[
G_{l,k}=(\mu_{l,k},\ell_{l,k}),
\qquad
\ell_{l,k}=\log\sigma_{l,k}.
\]

使用 log-standard-deviation 参数化可保证标准差为正，并使 prototype 插值更加稳定。

### 6.2 Style Router 输入

对于样本 \(Z_{i,l}\)，router 不直接使用普通 GAP，而使用显式风格统计：

\[
s_{i,l}
=\left[
\mu(Z_{i,l}),
\log\left(\sigma(Z_{i,l})+\epsilon\right)
\right].
\]

随后：

\[
\pi_{i,l}
=\operatorname{softmax}
\left(
f_l(\operatorname{sg}(s_{i,l}))/\tau_r
\right),
\]

其中：

- \(f_l\) 为两层 MLP；
- \(\operatorname{sg}\) 表示 stop-gradient；
- \(\tau_r\) 为路由温度，初始取 1.0。

停止 router 输入对 backbone 的梯度，可以减少 backbone 为满足路由正则而人为扭曲风格统计的退化风险。风格变换本身仍可向 backbone 传播梯度。

### 6.3 Sample-adaptive Golden Style

动态仿射参数为：

\[
\mu_{i,l}^{g}
=\sum_{k=1}^{K}\pi_{i,l,k}\mu_{l,k},
\]

\[
\ell_{i,l}^{g}
=\sum_{k=1}^{K}\pi_{i,l,k}\ell_{l,k},
\qquad
\sigma_{i,l}^{g}=\exp(\ell_{i,l}^{g}).
\]

然后执行：

\[
\tilde Z_{i,l}
=\sigma_{i,l}^{g}\odot
\frac{Z_{i,l}-\mu(Z_{i,l})}
{\sigma(Z_{i,l})+\epsilon}
+\mu_{i,l}^{g},
\]

\[
Z_{i,l}^{\mathrm{out}}
=(1-\alpha_l)Z_{i,l}
+\alpha_l\tilde Z_{i,l},
\qquad
\alpha_l=\operatorname{sigmoid}(a_l).
\]

这里的 \(\mu_{i,l}^{g},\sigma_{i,l}^{g}\) 应表述为自适应仿射风格参数，而不是混合高斯分布的精确均值和标准差。

### 6.4 Prototype 初始化

不能将所有 prototype 完全相同地初始化。推荐流程：

1. 在训练开始前，用前 5–10 个无标签 source/target batch 收集每层 \([\mu,\log\sigma]\)；
2. 每层独立执行 K-means；
3. 用聚类中心初始化 \((\mu_{l,k},\ell_{l,k})\)；
4. 后续将 prototype 作为可学习参数优化。

此过程不需要目标标签。

### 6.5 风格保持损失

保持变换前后的样本内容和二阶关系：

\[
\mathcal L_{\mathrm{pres}}^{l}
=\sum_{d\in\{s,t\}}
\Big(
\|\mu(Z_{d,l}^{\mathrm{out}})-\mu(Z_{d,l})\|_1
+\|\sigma(Z_{d,l}^{\mathrm{out}})-\sigma(Z_{d,l})\|_1
+\lambda_{\mathrm{gram}}
\|\operatorname{Gram}(Z_{d,l}^{\mathrm{out}})
-\operatorname{Gram}(Z_{d,l})\|_1
\Big).
\]

### 6.6 自适应 Golden Anchor 损失

源域样本全部参与：

\[
\mathcal L_{\mathrm{gold},s}^{l}
=\frac1{B_s}\sum_i
\left(
\|\mu(Z_{i,l}^{s,\mathrm{out}})-\mu_{i,l}^{g}\|_1
+\|\sigma(Z_{i,l}^{s,\mathrm{out}})-\sigma_{i,l}^{g}\|_1
\right).
\]

目标域样本按可靠度参与：

\[
\mathcal L_{\mathrm{gold},t}^{l}
=\frac{
\sum_j r_j
\left(
\|\mu(Z_{j,l}^{t,\mathrm{out}})-\mu_{j,l}^{g}\|_1
+\|\sigma(Z_{j,l}^{t,\mathrm{out}})-\sigma_{j,l}^{g}\|_1
\right)
}{\sum_jr_j+\epsilon}.
\]

这样，错误或含糊的目标预测不会与可靠样本等权地推动 style prototype。

### 6.7 类别条件 Routing Consistency

源域第 \(c\) 类的平均路由为：

\[
\bar\pi_{s,l}^{(c)}
=\frac{
\sum_i\mathbb I(y_i^s=c)\pi_{i,l}^s
}{
\sum_i\mathbb I(y_i^s=c)+\epsilon
}.
\]

目标域使用 teacher 软类别概率和可靠度：

\[
\bar\pi_{t,l}^{(c)}
=\frac{
\sum_j r_jq_{jc}\pi_{j,l}^t
}{
\sum_jr_jq_{jc}+\epsilon
}.
\]

类别条件路由损失为：

\[
\mathcal L_{\mathrm{route}}^{l}
=\frac1{|\mathcal C_l|}
\sum_{c\in\mathcal C_l}
D_{\mathrm{JS}}
\left(
\bar\pi_{s,l}^{(c)}
\|\bar\pi_{t,l}^{(c)}
\right),
\]

其中 \(\mathcal C_l\) 只包含当前 batch 中源域存在、且目标域有效软质量超过阈值的类别。

该设计按类别分别归一化，因此不会错误地强制类别先验不同的两个域拥有相同总体路由比例。

### 6.8 路由专一性与总体使用率

仅使用 uniform balance 会强制本应稀疏的 prototype 被平均使用。更合理的是同时约束：

样本级路由应具有一定专一性：

\[
\mathcal L_{\mathrm{sharp}}^{l}
=\frac1B\sum_i H(\pi_{i,l}).
\]

batch 级不能完全塌缩到单一 prototype：

\[
\mathcal L_{\mathrm{usage}}^{l}
=D_{\mathrm{KL}}
\left(
\bar\pi_l
\middle\|
\operatorname{Uniform}(K)
\right).
\]

两项权重都应较小，避免人为要求真实数据严格均匀地使用每个 prototype。

### 6.9 Prototype Separation

令：

\[
g_{l,k}=[\mu_{l,k},\ell_{l,k}],
\]

采用平滑排斥项：

\[
\mathcal L_{\mathrm{sep}}^{l}
=\frac{2}{K(K-1)}
\sum_{k<k'}
\exp\left(
-\frac{\|g_{l,k}-g_{l,k'}\|_2^2}{\tau_{\mathrm{sep}}}
\right).
\]

相比直接对原始 \((\mu,\sigma)\) 做 cosine margin，该形式不会因均值正负或标准差尺度不同而产生难以解释的相似度。

### 6.10 多原型风格总损失

\[
\mathcal L_{\mathrm{R\text{-}MPGA}}
=\sum_l
\Big[
\lambda_{p}\mathcal L_{\mathrm{pres}}^{l}
+\lambda_{g}
(\mathcal L_{\mathrm{gold},s}^{l}
+\mathcal L_{\mathrm{gold},t}^{l})
+\lambda_{r}\mathcal L_{\mathrm{route}}^{l}
+\lambda_{h}\mathcal L_{\mathrm{sharp}}^{l}
+\lambda_{u}\mathcal L_{\mathrm{usage}}^{l}
+\lambda_{s}\mathcal L_{\mathrm{sep}}^{l}
\Big].
\]

此外，原始跨域统计对齐不能将随机配对的 source/target 样本逐元素相减。若保留该项，应比较 batch 聚合统计或使用分布距离：

\[
\mathcal L_{\mathrm{batch\text{-}align}}^l
=\|\mathbb E_i[\mu_{i,l}^s]-\mathbb E_j[\mu_{j,l}^t]\|_1
+\|\mathbb E_i[\sigma_{i,l}^s]-\mathbb E_j[\sigma_{j,l}^t]\|_1.
\]

正式实现中建议先保留该 batch 统计版本，避免引入额外的 MMD/CORAL 超参数。

---

## 7. Reliability-aware Soft SGA

### 7.1 软语义亲和目标

将源域标签写成 one-hot 向量 \(y_i^s\)，将目标 teacher 概率记为 \(q_j\)。所有 teacher 输出均停止梯度。

Source-Source：

\[
A_{ij}^{SS}=(y_i^s)^Ty_j^s.
\]

Source-Target：

\[
A_{ij}^{ST}=(y_i^s)^Tq_j.
\]

Target-Target：

\[
A_{ij}^{TT}=q_i^Tq_j.
\]

在预测已校准、样本类别条件独立的近似下，点积可解释为两个样本同类的概率。

### 7.2 可靠目标节点门控

Soft affinity 不意味着所有目标节点都必须参与。首先定义候选集合：

\[
\mathcal T_c
=\left\{
j:\arg\max q_j=c,
\max q_j\ge\delta_p,
r_j\ge\delta_r
\right\}.
\]

推荐初值：

\[
\delta_p=0.70,
\qquad
\delta_r=0.20.
\]

为延续当前实现中的 anti-collapse 策略：

1. 只有三个类别都存在候选节点时才启用本 batch 的目标图监督；
2. 每类保留相同数量的最高 \(r_j\max(q_j)\) 样本；
3. 若条件不满足，仅训练 Source-Source 图边和源域 node loss。

后续可将跨 batch 类别队列作为增强实验，但不放入首版实现，以免同时引入过多变量。

### 7.3 边可靠度

对于通过门控的节点：

\[
R_{ij}^{SS}=1,
\]

\[
R_{ij}^{ST}=r_j,
\]

\[
R_{ij}^{TT}=r_ir_j.
\]

未通过门控的目标节点不参与边监督。

### 7.4 分块归一化的软边损失

对于图模块输出的 affinity logits \(\hat A_{ij}\)，定义每个分块：

\[
\mathcal L_b
=\frac{
\sum_{(i,j)\in\Omega_b}
R_{ij}
\operatorname{BCEWithLogits}
(\hat A_{ij},A_{ij}^{\mathrm{soft}})
}{
\sum_{(i,j)\in\Omega_b}R_{ij}+\epsilon
},
\]

其中：

\[
b\in\{SS,ST,TT\}.
\]

最终：

\[
\mathcal L_{\mathrm{edge}}^{U}
=\sum_{b\in\mathcal B_{\mathrm{valid}}}
\omega_b\mathcal L_b.
\]

首版推荐对当前有效分块等权平均。这样，目标候选数量变化不会让某一类分块仅因边数更多而支配总损失。

计算边损失时：

- 对称图只计算一次无序节点对，避免 \((i,j)\) 与 \((j,i)\) 重复计数；
- 排除对角线自边；
- GCN 如需自环，应在邻接传播阶段单独加入，而不是让恒为 1 的对角线降低训练损失。

### 7.5 稀疏图约束

稠密 target-target 图容易累积大量低信息软边。建议对预测 affinity 或语义 affinity 取 top-k 交集/并集，首版使用：

\[
k_g=8.
\]

为保证消融清晰，快速验证阶段可以先保持当前稠密实现；正式版本再增加稀疏图，并将其作为独立消融项。

### 7.6 Node Loss

图卷积后的 node classifier 首版仅使用源域真实标签：

\[
\mathcal L_{\mathrm{node}}
=\operatorname{CE}(h_i^s,y_i^s).
\]

不建议首版再添加目标伪标签 node CE，否则无法区分性能提升究竟来自软图监督还是普通自训练。

### 7.7 R-SoftSGA 总损失

\[
\mathcal L_{\mathrm{R\text{-}SoftSGA}}
=\lambda_e\mathcal L_{\mathrm{edge}}^{U}
+\lambda_v\mathcal L_{\mathrm{node}}.
\]

---

## 8. 总体优化目标

\[
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{cls}}
+\lambda_{\mathrm{dis}}\mathcal L_{\mathrm{dis}}
+\lambda_{\mathrm{style}}\mathcal L_{\mathrm{R\text{-}MPGA}}
+\gamma(t)
\mathcal L_{\mathrm{R\text{-}SoftSGA}}.
\]

其中：

- \(\mathcal L_{\mathrm{cls}}\)：源域监督分类损失；
- \(\mathcal L_{\mathrm{dis}}\)：域对抗损失；
- \(\gamma(t)\)：图适配 ramp-up 权重。

为避免域损失 clamp 后梯度长期为零，实验中还应记录达到 cap 的比例；若大量 iteration 被截断，应改为平滑饱和形式，而不是继续依赖硬 clamp。

---

## 9. 训练策略

### 9.1 训练前无标签初始化

使用少量 source/target batch 收集各层 style statistics，完成 K-means prototype 初始化。该阶段不更新参数、不访问目标标签。

### 9.2 Warm-up：Iteration 1–200

训练：

- 源域分类；
- 域对抗；
- R-MPGA；
- Source-Source edge loss；
- source node loss。

关闭所有包含目标节点的 ST/TT 图边损失：

\[
\gamma_{ST}(t)=\gamma_{TT}(t)=0.
\]

EMA teacher 正常更新并积累稳定预测。

### 9.3 Graph Ramp-up：Iteration 201–400

逐渐加入目标图监督：

\[
\gamma(t)
=\frac12
\left[
1-\cos\left(
\pi\frac{t-200}{200}
\right)
\right].
\]

### 9.4 Full Adaptation：Iteration 401–1000

使用完整损失，保持：

\[
\gamma(t)=1.
\]

每个 iteration 的顺序为：

1. teacher 对目标 batch 进行 \(M\) 次随机预测，得到 \(q_i,r_i\)；
2. student 前向；
3. 计算分类、域、风格和图损失；
4. 更新 student 与 graph module；
5. EMA 更新 teacher。

---

## 10. 推荐初始超参数

| 参数 | 快速验证 | 正式实验 |
|---|---:|---:|
| Prototype 数量 \(K\) | 3 | 1, 2, 3, 4 |
| Router hidden dim | 32 | 32 |
| Router temperature \(\tau_r\) | 1.0 | 0.7, 1.0 |
| MC Dropout 次数 \(M\) | 3 | 5 |
| Teacher EMA \(\rho\) | 0.99 | 0.99 |
| Reliability \(\beta\) | 2.0 | 1.0, 2.0 |
| 概率门槛 \(\delta_p\) | 0.70 | 0.70 |
| 可靠度门槛 \(\delta_r\) | 0.20 | 0.10, 0.20 |
| Graph warm-up | 200 iters | 200 iters |
| Graph ramp-up | 200 iters | 200 iters |
| \(\lambda_{\mathrm{route}}\) | 0.01 | 0.001, 0.01 |
| \(\lambda_{\mathrm{sharp}}\) | 0.001 | 0.001 |
| \(\lambda_{\mathrm{usage}}\) | 0.001 | 0.001 |
| \(\lambda_{\mathrm{sep}}\) | 0.001 | 0.001, 0.01 |
| Graph top-k \(k_g\) | 暂不启用 | 8 |

其余超参数首先保持 CAGA-SGA 基线不变。不能同时大范围搜索所有新参数；优先确定 R-SoftSGA 是否有效，再调 prototype 数量和路由正则。

---

## 11. 分阶段实现路线

### 阶段 0：基线冻结

目标：确认当前 CAGA-SGA 在两条迁移任务上可稳定复现，并保存：

- 每个目标被试的 accuracy、balanced accuracy、macro-F1；
- 每个 iteration 的三类原始伪标签数与实际入图数；
- classification、domain、style、edge、node loss；
- 预测类别分布。

在基线结果冻结前，不开始完整 MPUS-GA 实验。

### 阶段 1：Soft-SGA

仅将硬 affinity target 改为 teacher soft affinity：

- 保留现有 balanced selection；
- 暂令所有入选目标节点 \(r_i=1\)；
- 不引入多 prototype。

该阶段验证“软边目标”本身是否优于硬边。

### 阶段 2：R-SoftSGA

加入：

- EMA teacher；
- entropy + MI reliability；
- ST/TT 边可靠度；
- graph warm-up 和 ramp-up。

这是整个方案优先级最高、最可能稳定获益的部分。

### 阶段 3：MPGA

在原 hard-balanced SGA 上单独加入：

- style-statistics router；
- K-means prototype 初始化；
- log-standard-deviation mixture；
- 类别条件 routing consistency；
- sharp、usage、separation 正则。

该阶段隔离多原型风格对齐的独立作用。

### 阶段 4：完整 MPUS-GA

组合 R-MPGA 与 R-SoftSGA，使同一个 \(r_i\) 同时控制目标 style loss 和图边 loss。

### 阶段 5：稀疏图增强

仅在完整模型已经稳定提升后加入 top-k 语义图，避免将稀疏化与核心机制混淆。

---

## 12. 必做消融实验

### 12.1 Soft graph 与 reliability 的二因素消融

| Variant | EMA Teacher | Soft Edge | Entropy+MI Reliability |
|---|---:|---:|---:|
| CAGA-SGA-balanced | × | × | × |
| EMA-Hard-SGA | √ | × | × |
| Soft-SGA | √ | √ | × |
| R-Hard-SGA | √ | × | √ |
| R-SoftSGA | √ | √ | √ |

该表可以区分收益来自 teacher 平滑、soft target 还是 reliability weighting。

### 12.2 多原型消融

| Variant | Multi-Prototype | Conditional Route | Reliability-weighted Target Style |
|---|---:|---:|---:|
| Golden Style Baseline | × | × | × |
| MPGA-basic | √ | × | × |
| MPGA-route | √ | √ | × |
| R-MPGA | √ | √ | √ |

### 12.3 总体消融

| Model | R-MPGA | Soft Edge | Reliability |
|---|---:|---:|---:|
| CAGA-SGA-balanced | × | × | × |
| R-MPGA only | √ | × | √（仅 style） |
| R-SoftSGA only | × | √ | √（仅 graph） |
| MPUS-GA | √ | √ | √（共享） |

### 12.4 Prototype 数量

测试：

\[
K\in\{1,2,3,4\}.
\]

必须增加单元级退化验证：当 \(K=1\)、关闭 route/usage/separation loss，并复制原 Golden Style 参数时，新模块的前向结果应与原 StyleAlignLayer 在数值误差范围内一致。

不建议首轮测试 \(K=5\)，因为三分类、小 batch 和有限训练步数下，过多 prototype 更容易产生空 prototype 或不稳定路由。

### 12.5 可靠度组成消融

| Variant | Predictive Entropy | MC Mutual Information |
|---|---:|---:|
| Max probability | × | × |
| Entropy only | √ | × |
| MI only | × | √ |
| Entropy + MI | √ | √ |

---

## 13. 诊断与可解释性实验

### 13.1 可靠度是否真的识别错误伪标签

目标标签只能在训练完成后的离线分析中使用。报告：

- 正确和错误伪标签的平均可靠度；
- 可靠度检测伪标签错误的 AUROC；
- risk-coverage curve 与 AURC；
- teacher 概率的 ECE、Brier score；
- 不同可靠度分桶中的真实伪标签准确率。

若可靠度与真实正确性无明显相关性，则不能宣称 uncertainty 减少了图污染。

### 13.2 Prototype 是否发生塌缩

每层报告：

- prototype 平均使用比例；
- 样本级路由熵；
- prototype 两两距离；
- 每个 prototype 的 source/target 占比；
- 按情绪类别、源数据集和目标被试绘制路由热图。

理想结果不是所有 prototype 完全均匀，而是：

- 不存在长期零使用 prototype；
- prototype 之间有稳定差异；
- 同一情绪的 source/target 路由更接近；
- 路由不完全退化为情绪分类器。

### 13.3 图拓扑质量

训练后离线比较：

- hard SGA、Soft-SGA、R-SoftSGA 的邻接热图；
- 同类边与异类边的 affinity 分布；
- top-k 图边精度；
- source-target 和 target-target 分块的独立 edge loss；
- 错误伪标签节点的平均度与总边权。

如果 R-SoftSGA 有效，错误伪标签节点应具有更低的有效度数和总边权。

---

## 14. 实验协议与统计检验

### 14.1 主实验

在以下两条任务上分别完成全部 16 个目标被试：

- SEED-VII \(\rightarrow\) SEED-V；
- SEED-IV \(\rightarrow\) SEED-V。

报告：

- Accuracy；
- Balanced Accuracy；
- Macro-F1；
- 每个被试的详细结果；
- 均值、标准差和 95% bootstrap confidence interval。

### 14.2 随机种子

快速开发可使用一个固定种子。正式主结果至少使用三个 base seeds，例如：

\[
\{42,52,62\}.
\]

每个方法必须使用完全相同的数据划分、目标被试顺序和种子规则。

### 14.3 显著性检验

以目标被试为配对单位，对 MPUS-GA 与 CAGA-SGA-balanced 执行：

- 配对 Wilcoxon signed-rank test；
- 同时报告被试级差值的中位数和 bootstrap 置信区间；
- 不只报告 p-value，还报告平均提升和获益被试比例。

### 14.4 禁止的数据泄漏

目标标签不得用于：

- 超参数选择；
- 温度校准；
- early stopping；
- checkpoint 选择；
- prototype 数量选择；
- uncertainty threshold 选择。

目标标签只能用于固定预算结束后的最终指标和离线解释性分析。

---

## 15. 成功与失败判据

### 15.1 R-SoftSGA 值得继续的条件

满足以下大部分现象：

- 两条迁移任务的 balanced accuracy 或 macro-F1 均有一致提升；
- 提升不只来自少数目标被试；
- 错误伪标签的平均可靠度显著更低；
- ST/TT 图边损失更稳定，类别塌缩频率下降；
- 相比基线，计算代价仍可接受。

### 15.2 Multi-Prototype 值得保留的条件

- \(K>1\) 稳定优于 \(K=1\)；
- prototype 不塌缩；
- 至少两个 prototype 有持续、可解释的使用模式；
- 提升在两个迁移方向上不出现明显相反结论。

若只有 R-SoftSGA 稳定提升，而 Multi-Prototype 无效，应缩减论文为可靠软图方向，不应为了名称完整强行保留多原型模块。

---

## 16. 计算复杂度与实现注意事项

### 16.1 MC Dropout 开销

目标 teacher 前向增加约 \(M\) 倍目标分支推理成本，但不保存反向图。可将同一 target batch 复制 \(M\) 份后一次并行前向，但需要评估显存占用。

建议：

- 开发阶段 \(M=3\)；
- 正式实验 \(M=5\)；
- 记录 iteration time、峰值显存和总训练时间。

### 16.2 数值稳定性

- 所有 entropy 使用 \(q+\epsilon\)；
- 所有可靠度加权平均的分母使用 \(+\epsilon\)；
- MI 因浮点误差可能略小于 0，应 clamp；
- prototype 标准差在 log space 参数化；
- 当某个图分块没有有效边时，跳过该分块而不是对空 tensor 计算 BCE；
- 当没有目标节点通过门控时，graph loss 仍保留 SS edge 和 source node 两部分。

### 16.3 梯度边界

以下量必须停止梯度：

- teacher 输出 \(p_i^{(m)}\)、\(q_i\)、\(r_i\)；
- soft affinity targets；
- router 的 style-statistics 输入建议 stop-gradient。

student feature、student affinity logits、style prototype 和 router 参数保持可训练。

---

## 17. 与当前代码的对应修改点

### `golden_style.py`

- 将 `GoldenStyleBank` 扩展为 `[num_layers, K, d_model]`；
- 新增每层 Style Router；
- 使用 `[mu, log_std]` 作为 router 输入；
- 输出逐样本 `mu_g`、`sig_g` 和 routing weights；
- 增加 K-means bootstrap 接口；
- 保留 `K=1` 退化路径。

### `model.py`

- StyleAlignLayer 接受 `[B, D]` 的逐样本 Golden Style；
- `style_infos` 额外返回 routing weights 和逐样本 style target；
- 支持 teacher 随机推理；
- 不在模型内部更新 EMA。

### `graph_align.py`

- 图模块本身可以保持输出 affinity logits 和 node logits；
- 可选增加 top-k 邻接稀疏化；
- 自环与 edge supervision mask 分开处理。

### `main.py`

- 创建无梯度 EMA teacher；
- 实现 MC teacher prediction；
- 实现 entropy、MI 和 reliability；
- 将 hard ideal affinity 替换为分块 soft affinity；
- 实现类别平衡节点门控、可靠度边权和空分块保护；
- 增加 warm-up/ramp-up；
- 记录 uncertainty、prototype usage 和各图分块指标；
- 将所有新增机制做成独立 CLI 开关，便于消融。

建议新增的主要参数：

```text
--num-style-prototypes
--router-hidden-dim
--router-temperature
--teacher-ema
--mc-passes
--reliability-beta
--min-soft-confidence
--min-reliability
--graph-warmup-iters
--graph-rampup-iters
--soft-graph
--uncertainty-weighting
--graph-topk
```

---

## 18. 论文贡献点的建议表述

### Contribution 1

提出 reliability-aware multi-prototype golden alignment，将单一全局风格锚点扩展为由显式 EEG 风格统计路由的自适应 prototype 空间，并通过类别条件路由对齐处理跨数据集类别先验不一致问题。

### Contribution 2

提出 reliability-aware soft semantic graph，以 EMA teacher 的概率内积表示跨域同类概率，并联合预测熵与 MC 模型分歧控制目标节点和边的监督强度，降低稳定错误伪标签造成的图结构污染。

### Contribution 3

建立统一的可靠度驱动机制，使目标样本可靠度同时调节风格 prototype 学习和语义图监督，从而将特征统计对齐与语义拓扑对齐整合为一个协同的跨数据集 EEG 域适配框架。

论文中不宜声称多原型、MC Dropout 或软标签本身是首次提出；创新重点应放在：

- EEG 跨数据集场景中的类别条件 style routing；
- 可靠度对 style 与 graph 两个对齐层面的统一控制；
- 对错误目标信息传播路径的系统性建模与验证。

---

## 19. 最小可行版本

如果计算资源或时间有限，优先实现以下 MVP：

1. EMA teacher；
2. \(M=3\) 的 entropy + MI reliability；
3. balanced node gate；
4. soft ST/TT affinity；
5. reliability-weighted、分块归一化 edge loss；
6. 200 iteration warm-up + 200 iteration ramp-up。

MVP 暂不实现：

- Multi-Prototype Golden Style；
- graph top-k；
- target node pseudo-label loss；
- 跨 batch memory bank。

只有当 MVP 相比 CAGA-SGA-balanced 在两条迁移任务上呈现稳定收益后，再实现 R-MPGA。该顺序可以用最低成本判断本方案最核心的“可靠软图”假设是否成立。

---

## 20. 最终方法命名

完整模型保留：

**MPUS-GA**<br>
**Multi-Prototype Uncertainty-aware Style and Graph Alignment**

中文可表述为：

**多原型可靠度驱动的风格—语义图协同对齐网络**。

其中“uncertainty-aware”在正文中统一落到可计算的 \(r_i\)，避免将 uncertainty 仅作为概念性描述。
