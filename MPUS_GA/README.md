# MPUS-GA multiscale DE project

This subproject is isolated from the CAGA-SGA reproduction code. It rebuilds
SEED-IV, SEED-V, and SEED-VII differential-entropy features from trial EEG at
three non-overlapping window lengths: 1 s, 2 s, and 4 s.

## Project layout

```text
MPUS_GA/
├── preprocessing/       # raw CNT -> multiscale DE and artifact validation
├── protocols/           # shared fixed comparison protocol
├── trial_temporal/      # multiscale data, model, losses, and training entry point
├── scripts/             # nohup-friendly experiment launcher
├── tests/               # preprocessing and trial-model tests
├── data_processed/      # generated 1 s / 2 s / 4 s NPZ files
├── results_bidirectional_full_ablation/   # paired A/B ablations and Main
└── logs/bidirectional_full_ablation/      # 16 experiment logs + suite.log
```

The original CAGA-SGA implementation remains in the repository root and is not
imported or moved by this subproject.

## Data policy

- Raw EEG is segmented at trial boundaries before feature extraction.
- Every scale is computed directly from raw trial EEG. A 2 s or 4 s feature is
  never produced by averaging shorter DE windows.
- No LDS or moving-average smoothing is applied across windows.
- Outputs are unnormalised DE values. Source/target normalisation must be fitted
  later inside each transfer fold to avoid leakage.
- Both the original emotion label and the CAGA-SGA three-class mapping are
  retained.
- Recordings without recoverable trial boundaries are never segmented using
  guessed timestamps. They are listed in `seed_vii/manifest.json` under
  `excluded_files`; use `--missing-trigger-policy error` for strict failure.
- SEED-IV MAT files already contain 24 trial-segmented 62-channel arrays at
  200 Hz. They are not resampled or segmented again. The single endpoint sample
  left after complete-second windows is discarded with the normal window
  remainder.

Three-class mapping:

| Class | Emotions |
|---|---|
| 0, positive | happy, surprise |
| 1, neutral | neutral |
| 2, negative | disgust, fear, sad, anger |

## Expected source layout

The default dataset root is inherited from the parent project:

```text
/dataset/gzw/seed_series/
├── eeg_raw/SEED_IV/{1,2,3}/*.mat
├── eeg_raw/SEED_VII/*.cnt
├── eeg_raw/SEED_V/*.cnt
├── labels/SEED_IV/Channel Order.xlsx
└── labels/SEED_VII/emotion_label_and_stimuli_order.xlsx
```

Several legacy path spellings are detected automatically. Explicit paths can
be supplied when the server layout differs.

## Output layout

```text
data_processed/
├── processing_config.json
├── processing_summary.json
├── validation_report.json
├── processing_runs/       # immutable per-run records
├── seed_iv/
│   ├── manifest.json
│   ├── window_1s/subject_01_session_1.npz
│   ├── window_2s/subject_01_session_1.npz
│   └── window_4s/subject_01_session_1.npz
├── seed_vii/
│   ├── manifest.json
│   ├── window_1s/subject_01_session_1.npz
│   ├── window_2s/subject_01_session_1.npz
│   └── window_4s/subject_01_session_1.npz
└── seed_v/
    ├── manifest.json
    ├── window_1s/subject_01_session_1.npz
    ├── window_2s/subject_01_session_1.npz
    └── window_4s/subject_01_session_1.npz
```

Each NPZ stores:

- `features`: `[N, 62, 5]` float32 DE features;
- `label_original`: original seven-class or five-class IDs;
- `label_3class`: positive/neutral/negative IDs;
- `subject_id`, `session_id`, `trial_id`, `session_trial_id`;
- `window_id`, `start_second`;
- `channel_names`, band definitions, sampling rate, and source filename.

## Run

The server BCI environment already contains MNE, NumPy, SciPy, and openpyxl:

```bash
CUDA_VISIBLE_DEVICES=1 MNE_DONTWRITE_HOME=true \
  /home/gzw/anaconda3/envs/BCI/bin/python -m \
  MPUS_GA.preprocessing.preprocess \
  --data-root /dataset/gzw/seed_series \
  --datasets seed-vii seed-v \
  --window-seconds 1 2 4
```

Use explicit paths if automatic discovery does not match the server:

```bash
MNE_DONTWRITE_HOME=true /home/gzw/anaconda3/envs/BCI/bin/python -m \
  MPUS_GA.preprocessing.preprocess \
  --seed-vii-raw-dir /path/to/SEED_VII/EEG_raw \
  --seed-v-raw-dir /path/to/SEED_V/EEG_raw \
  --seed-vii-label-file /path/to/emotion_label_and_stimuli_order.xlsx
```

For a small pipeline check:

```bash
MNE_DONTWRITE_HOME=true /home/gzw/anaconda3/envs/BCI/bin/python -m \
  MPUS_GA.preprocessing.preprocess \
  --datasets seed-vii --subjects 1 --window-seconds 1 2 4
```

Validate all generated files:

```bash
/home/gzw/anaconda3/envs/BCI/bin/python -m \
  MPUS_GA.preprocessing.validate_processed
```

Process all SEED-IV subjects in the background from the repository root:

```bash
mkdir -p MPUS_GA/logs/preprocessing
RUN_LOG="MPUS_GA/logs/preprocessing/seed_iv_$(date +%Y%m%d_%H%M%S).log"
nohup bash MPUS_GA/scripts/run_seed_iv_preprocessing.sh > "${RUN_LOG}" 2>&1 &
echo "PID=$! LOG=${RUN_LOG}"
```

SEED-IV contributes 45 subject-sessions and 135 NPZ artifacts. Its original
labels are neutral=0, sad=1, fear=2, and happy=3. The retained three-class view
maps happy to positive, neutral to neutral, and sad/fear to negative.

`processing_config.json` describes the stable feature pipeline and does not
claim a subject subset. Each invocation is recorded separately under
`processing_runs/`; dataset manifests and `processing_summary.json` inventory
all NPZ artifacts present on disk rather than only the most recent batch.

## Multi-source multiscale training

`trial_temporal/train.py` is the canonical training entry point. It aligns the
1 s, 2 s, and 4 s windows of each trial. The A-series uses SEED-VII and/or
SEED-IV as labeled sources and one SEED-V subject as the unlabeled target. B6
reverses the direction, using all SEED-V trials as the labeled source and one
SEED-VII subject as the unlabeled target.

### Fixed transductive UDA experiment paradigm

The experiment is **transductive unsupervised domain adaptation**, not a
conventional supervised train/test split. Source and target data have different
roles:

| Experiment | Labeled source pool | Unlabeled target fold | Independent folds |
|---|---|---|---:|
| A6, SEED-VII -> SEED-V | all 20 SEED-VII subjects, 1600 trials | all 45 trials of one SEED-V subject | 3 seeds x 16 subjects = 48 |
| B6, SEED-V -> SEED-VII | all 16 SEED-V subjects, 720 trials | all 80 trials of one SEED-VII subject | 3 seeds x 20 subjects = 60 |

There is no held-out subset inside the source dataset. The source sampler draws
with replacement from the complete labeled source pool, with softened
inverse-frequency strength `source_balance_alpha=0.4`. There is also no split
of the current target subject into separate adaptation and evaluation EEG
subsets. The complete target-subject EEG pool is sampled with replacement for
unlabeled adaptation, and the same complete pool is traversed once for final
evaluation. The adaptation view does not return `y`; target labels cannot enter
normalization, sampling, pseudo-label construction, losses, gradients, or model
selection.

Every `(random seed, target subject)` pair initializes and trains a fresh model
for exactly 1000 iterations. There is no validation phase, early stopping, or
checkpoint/epoch selection. Target labels are read exactly once, after
iteration 1000, to report raw fused-logit trial Accuracy, Balanced Accuracy,
Macro-F1, the confusion matrix, and per-class recall. The evaluated model is
therefore always the fixed final model, even when an earlier iteration might
have performed better.

This is deliberately stricter than the public original CAGA-SGA `main.py`,
which evaluates the same target `test_dataset` after every epoch and retains
the epoch with the highest target-test Accuracy. The modified CAGA-SGA baseline
in this repository and MPUS-GA instead share
`protocols/fixed_transductive_uda.py`: fixed 1000 iterations, no target-label
checkpoint selection, and one final target evaluation. Results from the public
target-selected protocol must not be presented as if they used this fixed-final
protocol.

The prediction path is:

```text
aligned 1 s / 2 s / 4 s DE sequences [T,62,5]
  -> EEG channel attention and shared dynamic graph encoder
  -> scale-specific temporal convolution and Transformer encoders
  -> independent per-scale trial embeddings and classification logits
  -> source scale-class reliability and a learnable 3 x 3 residual matrix
  -> sample-dependent weights for each scale x emotion-class edge
  -> one weighted pyramid feature for each emotion class
  -> stable residual with the relation-weighted logit fusion
  -> source-excess and cross-scale boundary-attractor suppression
  -> one final-logit trial prediction
```

Per-scale logits are computed before the cross-scale context Transformer, so a
scale prediction cannot observe another scale's input. The contextualized
scale embeddings are used only to determine fusion weights and the fused
embedding. Consequently, per-scale metrics are genuine independent-scale
evidence, and their comparison with the fused output is a valid multiscale
ablation.

### Class-conditional weighted feature pyramid

The association between the three temporal scales and three emotion classes is
represented by a `[3 scales, 3 classes]` bipartite relation matrix. Source
class-wise margins initialize the relation. A zero-initialized learnable 3 x 3
residual and a sample-dependent gate refine it, while a uniform weight floor
prevents a scale from disappearing. Every class column is normalized over the
1 s, 2 s, and 4 s levels.

The relation matrix now controls features rather than merely averaging logits.
The model projects each contextual scale level, constructs a separate weighted
pyramid feature for positive, neutral, and negative, and scores each feature
with the matching classifier row. The established 3 x 3 relation-weighted
logits remain the stable anchor. Legacy Main uses its original scalar residual;
the safety-gated G family below replaces that scalar only inside the new G
experiments.

### Safety-gated class-conditional pyramid (G family)

The G experiments test whether the feature pyramid can add useful multiscale
information without destroying the stronger relation-logit anchor. For class
`c` of target trial `i`, the final logit is

```text
z(i,c) = z_anchor(i,c) + gate(i,c) * clip(z_pyramid(i,c)-z_anchor(i,c), -0.5, 0.5)
```

The gate starts at 0.05. It is forced to zero through iteration 300, then uses
the same 300-to-600 ramp as domain adaptation. Thus the source classifier and
independent scale heads are established before any pyramid correction can
change a prediction. A detached source-label teacher opens a class gate when
the pyramid raises the true-class logit or lowers a false-class logit; a small
sparsity penalty keeps the anchor as the default. The teacher supervises only
the gate and cannot feed labels into pyramid features or scale heads.

The sample-class gate uses six pieces of detached evidence for each class:
mean independent-scale probability, disagreement, anchor probability,
relation concentration, cross-scale agreement, and the proposed pyramid-logit
change. G3 additionally closes a target class gate when the pyramid proposes an
increase unsupported by the independent scale heads, or when the existing
unlabeled cross-scale detector reports shared source-excess/boundary-attractor
risk. This guard is one-sided: it suppresses risky pyramid corrections but does
not manufacture support for another class. No target labels, target-prior
correction, new data split, or new preprocessing are introduced.

An online target-prior estimator uses only raw independent-scale logits. It
maintains a soft source confusion matrix for every scale and matches all target
scale mean probabilities jointly with a constrained ridge label-shift solve.
The estimated target prior is floored and normalized. In the current method its
correction strength defaults to zero: the estimate is diagnostic-only and
cannot change final logits, pseudo-labels, domain alignment, or prototypes.
This isolation is intentional because direct label-shift inversion was unstable
under cross-dataset conditional shift. A nonzero correction remains an explicit
experimental option rather than part of the paired Main defaults.

The same source soft-confusion statistics provide two one-sided cross-scale
bias signals. The first compares target soft-probability and hard-decision
rates against their natural-source references. The second detects a
decision-boundary attractor: a scale chooses a class as argmax substantially
more often than the total soft-probability mass assigned to that class. Its
per-scale signal is
`relu(log(hard_frequency / mean_probability) - log(1.15))`. Taking the median
over 1 s, 2 s, and 4 s means at least two independent scale heads must report
the same class-specific problem. A legitimate confident target-prior shift,
where hard frequency and probability mass rise together, does not trigger this
second signal.

Both signals are multiplied by the combined soft/hard source false-positive
risk, start after iteration 300, follow the adaptation ramp, and share a total
cap of 0.5 logit. They can only suppress a suspect class and can never boost
one. The source-excess and boundary-attractor strengths both default to 2.0;
their relative tolerances are 10% and 15%, respectively.

Training uses the target evidence EMA available from the preceding update. At
every reported evaluation, the final model makes an extra deterministic pass
over the complete `UnlabeledMultiScaleView` and recomputes raw independent
scale probability means and hard frequencies. This fresh snapshot is used only
for the boundary-attractor term; the longer-horizon source-excess term retains
its training EMA. The refresh loader cannot contain `y`, so this closes the gap
between training-time EMA and final-classifier drift without using target
labels for calibration.

Target pseudo-label evidence is constructed from the adjusted but still
independent per-scale probabilities. A target trial is eligible only when its
consensus confidence, scale vote count, and Jensen-Shannon agreement pass their
thresholds. These detached consensus labels update scale/class target
prototypes after iteration 300. Neither fused pyramid logits nor target labels
can enter the common-bias detector, prior estimator, or pseudo-label
construction.

The hard issue-7 boundary is therefore:

```text
each scale input -> its own embedding -> its own logits
                                        |
all independent evidence ---------------+
  -> source-excess + hard/soft boundary-attractor detection
  -> diagnostic target-prior estimate + final unlabeled evidence refresh
  -> bounded class suppression + independent-scale consensus
  -> source relation + learnable 3 x 3 residual + sample importance
  -> positive / neutral / negative weighted pyramid features
  -> relation-weighted logits + small pyramid residual -> final logits
```

Neither cross-scale context nor relation-graph output can feed back into the
forward computation of a per-scale embedding or logit. Target pseudo-labels
also cannot be obtained from relation-guided fused logits. Regression tests
perturb other-scale inputs and the relation matrix independently and require
every unaffected per-scale embedding/logit to remain identical.

The source objective adds mean independent-scale classification loss with
weight 0.3 and a source-label scale-gate supervision loss with weight 0.1.
Conditional domain and prototype losses use class-wise means before averaging
over valid pseudo classes, so the negative majority cannot dominate adaptation
through sample count. This balances optimization contributions without
forcing the target prediction prior to be uniform.

The complete bidirectional fixed-final suite uses isolated paths:

```text
results_bidirectional_full_ablation/{A0,B0,...,A6,B6,A_main,B_main}/
logs/bidirectional_full_ablation/{A0,B0,...,A6,B6,A_main,B_main}.log
logs/bidirectional_full_ablation/suite.log
```

Result JSON files record the source relation matrix, learned 3 x 3 residual,
sample gate, source prediction reference and precision for every scale/class,
three-scale training EMA and final unlabeled target evidence, separate
source-excess and boundary-attractor adjustments, their bounded total logit
adjustment, estimated target prior, pyramid residual weight, per-class target
pseudo-label counts, worst-class recall, and recall gap. Raw and adjusted
per-scale metrics are both retained.

The full method maintains EMA prototypes for every source-domain/scale/class
combination from source truth labels and scale/class target prototypes from
high-confidence, unlabeled target predictions. Their cosine agreement controls
which source and which temporal scale is trusted for each class. The same
structure drives per-scale conditional domain alignment, differentiable
prototype alignment, and class-dependent multiscale fusion.

Source prototypes may accumulate during the source-only warmup, but target
prototypes, target-derived reliability, and prototype-alignment loss are all
disabled through iteration 300. Target prototype initialization begins at
iteration 301, after the classifier has completed the source-only adaptation
warmup. This prevents biased early target predictions from changing later
fusion weights during warmup.

The Main final prediction is the argmax of the relation-weighted logits plus the
small pyramid residual and the jointly bounded source-excess/boundary-attractor
suppression. No target-prior correction is applied by default. Neither bias
detector, the final evidence refresh, nor the diagnostic prior estimator
accesses target labels. The old
supervised contrastive, InfoMax, and 1-second-teacher consistency losses are not
part of this experiment family. Source inverse-frequency sampling remains
softened to alpha=0.4; domain and prototype adaptation start after iteration
300 and reach full weight at iteration 600.

The comparison protocol remains fixed:

- every configured source dataset contributes its complete labeled trial pool;
- all trials of the current target subject form the unlabeled target domain;
- training runs for exactly 1000 iterations without validation or checkpoint
  selection;
- target labels are accessed once for final trial-level evaluation;
- Main final predictions use bounded source-excess and boundary-attractor
  suppression; each ablation records which component is disabled, and none
  uses target-prior correction;
- random seeds 42, 43, and 44 and all target subjects run by default.

### CAGA-SGA-style target-selected comparison

The optional `--evaluation-protocol caga_target_best` mode reproduces the
model-selection paradigm of the public CAGA-SGA code while leaving this
project's model, data, losses, optimizer, and 1000-iteration training budget
unchanged. It evaluates the complete labeled target subject every 50
iterations by default and reports the candidate with the highest target
Accuracy. A tie retains the earlier candidate, matching CAGA-SGA's strict
`val_acc > best_val_acc` update rule.

This mode deliberately uses target labels for model selection. They still do
not enter the adaptation batches, losses, or gradients, but the reported score
is not a label-blind fixed-final UDA result. Each result JSON records the full
target-evaluation trace, the selected iteration, selection criterion, and
number of target evaluations. Use an isolated result root so these files do
not overwrite fixed-final results.

For a controlled target-selected comparison, use the paired Main tags and
change only the evaluation protocol:

```text
results_caga_target_selected/A_main/
results_caga_target_selected/B_main/
logs/caga_target_selected/A_main.log
logs/caga_target_selected/B_main.log
```

The formal experiment matrix is:

Every `A`/`B` pair has identical architecture, losses, and hyperparameters;
only the transfer direction changes. `A` means SEED-VII to SEED-V and `B`
means SEED-V to SEED-VII.

| Paired tags | Purpose |
|---|---|
| A0 / B0 | 1 s single-scale baseline; cross-scale suppressors are inapplicable |
| A1 / B1 | 2 s single-scale baseline; cross-scale suppressors are inapplicable |
| A2 / B2 | 4 s single-scale baseline; cross-scale suppressors are inapplicable |
| A3 / B3 | uniform 1/2/4 s fusion instead of class-conditional reliability fusion |
| A4 / B4 | relation-weighted logit fusion with the feature pyramid genuinely absent |
| A5 / B5 | source-excess suppression retained, boundary-attractor suppression removed |
| A6 / B6 | both source-excess and boundary-attractor suppression removed |
| A_main / B_main | complete boundary-reliable class-conditional weighted pyramid |

The isolated G matrix keeps the same fixed-final protocol and both bias
suppressors in every row; only the way the pyramid is admitted changes:

| Paired tags | Purpose |
|---|---|
| A_G0 / B_G0 | relation-logit anchor with no feature pyramid; equivalent architectural control to A4 / B4 |
| A_G1 / B_G1 | static learned gate for each emotion class, with delayed activation and bounded residual |
| A_G2 / B_G2 | sample-by-class dynamic gate using detached independent-scale evidence |
| A_G3 / B_G3 | G2 plus the unlabeled common-bias and unsupported-pyramid safety guard |

### Class-query low-rank multiview fusion (H family)

The H family treats 1 s, 2 s, and 4 s as three temporal views while retaining
their independent embeddings and logits. H2 and later construct six downstream
fusion tokens: `1s`, `2s`, `4s`, `1sx2s`, `1sx4s`, and `2sx4s`. Each pair token
uses a rank-16 multiplicative interaction rather than a full tensor product.
Three learned emotion-class queries separately attend to these tokens, so each
class can use different single-scale and cross-scale evidence.

H3 subtracts detached predictive entropy and class-wise cross-scale conflict
from the attention scores. These reliability terms can only change downstream
fusion and cannot alter an independent scale head. H4 uses the existing
unlabeled common-bias evidence as a sign-aware guard: a risky class has an
unsupported positive logit correction attenuated, while a negative correction
that lowers the same over-predicted class remains available. The relation-logit
path stays the stable anchor, the correction remains clipped to `[-0.5, 0.5]`,
and the pyramid is disabled through iteration 300 before ramping to full weight
at iteration 600.

H5 addresses the cross-seed instability observed in H4. The EMA 3 x 3
scale-by-class relation estimated from source labels and independent scale
logits is expanded to the six multiview tokens; pair-token reliability is the
geometric mean of its two scales. Half of the dynamic attention is shrunk to
this source-only anchor. Target pseudo-label prototypes are deliberately
excluded from the anchor. The sample-class pyramid gate is also shrunk halfway
toward its class-static prior and passed through a smooth `0.25` ceiling, so a
seed-specific gate cannot overwhelm the relation-logit path.

| Paired tags | Purpose |
|---|---|
| A_H0 / B_H0 | relation-logit anchor without feature fusion; same architectural control as A4 / B4 |
| A_H1 / B_H1 | three class queries attend to the three scale tokens |
| A_H2 / B_H2 | H1 plus three rank-16 pairwise scale-interaction tokens |
| A_H3 / B_H3 | H2 plus entropy- and conflict-aware attention |
| A_H4 / B_H4 | H3 plus sign-aware protection against harmful class-logit increases |
| A_H5 / B_H5 | H4 plus source-only reliability anchoring and stable bounded pyramid admission |

Result JSON files include the six token names, class-by-token mean attention,
the pre-shrink dynamic attention, the source-only token anchor, token
uncertainty and class-wise conflict, as well as the existing residual gate and
guard diagnostics. No new preprocessing, target labels, target-prior
correction, or target checkpoint selection is introduced.

### Relative-degradation-aware temporal pyramid (R family)

The R family translates the two useful ideas in RDANet to trial EEG without
copying its image-specific pixel rearrangement or Fourier phase operations.
R1 adds a one-dimensional temporal MSAD path after the independent heads:
fixed normalized binomial filters of widths 3, 5, and 7 are mixed per channel,
folded by adjacent time pairs, refined depthwise, and injected only into the
downstream 2 s/4 s fusion context. It therefore cannot contaminate the native
1 s/2 s/4 s logits.

R2 adds a source-only multi-slot memory with shape
`[3 scales, 3 classes, K slots, d]`. Every scale-class cell receives exactly
the same number of slots. The memory is updated only from source embeddings
and source true labels through iteration 300, then frozen. A target trial
queries every class independently; no target pseudo-label selects the queried
class and target data never update a slot. R3 uses cosine distance to those
retrieved source prototypes as relative degradation: degraded scale-class and
pairwise tokens are downweighted before class-query attention, and the same
distance controls how source-memory features are fused across scales.

| Paired tags | Purpose |
|---|---|
| A_R0 / B_R0 | exact H5 stable source-anchored multiview baseline |
| A_R1 / B_R1 | R0 plus anti-aliased temporal MSAD |
| A_R2 / B_R2 | R1 plus equal-capacity source class multi-prototype memory |
| A_R3 / B_R3 | R2 plus relative-degradation-aware class-conditional fusion |
| A_R4 / B_R4 | H5 plus source class multi-prototype memory only; MSAD and relative degradation removed |

The R variants use the existing processed 1 s/2 s/4 s artifacts and the same
fixed-final protocol. Result files record filter mixtures, MSAD admission
gates, memory initialization/update counts, class-wise prototype distance,
memory scale weights, and memory residual gates.

All optional R modules are initialized after the complete shared H5 model.
Consequently, resetting the same random seed produces byte-identical shared
H5/R weights, so an ablation cannot gain merely by shifting the random-number
sequence used for downstream shared layers.

### Brain-region-guided electrode topology prior (P family)

The P family keeps R2 unchanged and adds a soft sensor-space topology bias to
the dynamic 62-electrode graph. This is intentionally described as an
electrode prior rather than source-localized cortical connectivity. The fixed
matrix combines four local montage neighbours, seven coarse scalp regions,
bilateral homologous electrode pairs, and self-connections. It is injected
into the graph score before top-k selection; the union of dynamic and prior
candidates remains available, so the prior never acts as a hard mask.

The same 62-channel order is validated when every processed NPZ is loaded.
P2 learns one bounded prior strength shared by 1/2/4 s, while P3 learns one
strength per temporal scale. These constant-initialized gates consume no
random numbers, preserving R2 initialization. P4 applies a deterministic
channel permutation to the same matrix as a matched negative control.

| Paired tags | Purpose |
|---|---|
| A_P0 / B_P0 | exact R2 control with no topology prior |
| A_P1 / B_P1 | fixed soft electrode-topology prior |
| A_P2 / B_P2 | learned topology strength shared across scales |
| A_P3 / B_P3 | learned topology strength for each of 1/2/4 s |
| A_P4 / B_P4 | P3 with a permuted-topology negative control |

No target labels, target pseudo-label routing, raw-signal reprocessing, or
target checkpoint selection is introduced. Result JSON files record the
topology mode, canonical channel order, region names, and final effective
strength for each scale.

For a two-direction seed-42 pilot, start in `/home/gzw/projects`, use isolated
output paths, and assign one physical GPU to each direction:

```bash
mkdir -p MPUS_GA/results_safe_pyramid_g_seed42 \
  MPUS_GA/logs/safe_pyramid_g_seed42
CUDA_VISIBLE_DEVICES=0 nohup /home/gzw/miniforge3/envs/BCI/bin/python -m \
  MPUS_GA.trial_temporal.train --experiment A_G3 --random-seeds 42 \
  --target-subjects all --evaluation-protocol fixed_final \
  --result-root MPUS_GA/results_safe_pyramid_g_seed42 \
  --device cuda:0 > MPUS_GA/logs/safe_pyramid_g_seed42/A_G3.log 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 nohup /home/gzw/miniforge3/envs/BCI/bin/python -m \
  MPUS_GA.trial_temporal.train --experiment B_G3 --random-seeds 42 \
  --target-subjects all --evaluation-protocol fixed_final \
  --result-root MPUS_GA/results_safe_pyramid_g_seed42 \
  --device cuda:0 > MPUS_GA/logs/safe_pyramid_g_seed42/B_G3.log 2>&1 < /dev/null &
```

On CSU, start the complete dual-GPU suite from the `MPUS_GA` directory:

```bash
bash scripts/run_bidirectional_full_ablation_suite.sh
```

The controller detaches itself with `nohup` and maintains a global FIFO order:
`A0, B0, A1, B1, ..., A6, B6, A_main, B_main`. Physical GPUs 0 and 1 each
receive one experiment initially. Whenever either experiment finishes, that
same GPU receives the next queued experiment after three seconds. Thus at most
two experiments run concurrently, a faster GPU never waits for the other
queue, and there is no GPU-idle polling.

The script creates 16 result directories, 16 experiment logs, and one
`suite.log`. The suite log records each experiment ID, start time, end time,
and physical GPU. It writes no PID, status, or JSONL side files. A new launch
overwrites these 17 logs; completed result JSON files are skipped unless
`OVERWRITE=1` is set.

Override the default fold set when doing a smoke run, for example:

```bash
TARGET_SUBJECTS=1 RANDOM_SEEDS="42" \
  bash scripts/run_bidirectional_full_ablation_suite.sh
```

With the defaults, every A experiment contains 3 seeds x 16 subjects = 48
folds, and every B experiment contains 3 seeds x 20 subjects = 60 folds. The
complete suite therefore produces 864 fold JSON files. Source normalization is
always fitted only on the configured source dataset.

### Paired statistical analysis

Analyze the completed suite with the target subject, rather than an individual
seed/fold, as the independent statistical unit:

```bash
/home/gzw/miniforge3/envs/BCI/bin/python \
  scripts/analyze_bidirectional_ablation.py \
  --result-root results_bidirectional_full_ablation \
  --output-dir analysis/bidirectional_full_ablation
```

The script validates all 864 JSON files as fixed-final runs, averages the three
seeds inside each target subject, and then performs the planned paired
comparisons. It writes subject-level metrics, paired mean differences with 95%
bootstrap confidence intervals, two-sided Wilcoxon tests, per-metric Holm
correction, Cohen's paired `dz`, and matched-pairs rank-biserial effects. A
positive reported performance change is always favorable; recall-gap changes
are sign-flipped so that a reduction is positive. Prediction-rate columns are
descriptive raw candidate-minus-reference changes because a larger predicted
class share is not inherently better.

## Signal processing

1. Drop non-EEG channels from `{M1, M2, ECG, HEO, VEO}` when present.
2. Broadband filter at 0.1–70 Hz and apply a 50 Hz notch filter.
3. Resample to 200 Hz.
4. Cut trials using SEED-VII event pairs or SEED-V session timestamps.
5. For each trial, apply fourth-order zero-phase Butterworth filters to:
   delta 1–4 Hz, theta 4–8 Hz, alpha 8–14 Hz, beta 14–31 Hz, and gamma
   31–50 Hz.
6. Calculate Gaussian differential entropy from the within-window variance:

   `DE = 0.5 * ln(2 * pi * e * (variance + epsilon))`.

The zero-phase filter is appropriate for the intended offline transfer
experiment. It should be replaced by a causal filter if the model is later
evaluated as a real-time decoder.

## Known source-data exceptions

`14_20221015_1.cnt` contains no embedded trial-boundary annotations, so the
preprocessor uses the official manual trigger information discovered at
`labels/SEED_VII/save_info.zip`. `9_20221111_3.cnt` contains two unrelated
keypad annotations; the preprocessing code filters those out and retains its
40 valid trial-boundary events. Thus all 80 SEED-VII sessions are recoverable
with the current server dataset.
