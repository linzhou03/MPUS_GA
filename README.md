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
├── results_class_conditional_multiscale/  # exactly A0..A7 and main
└── logs/class_conditional_multiscale/     # fixed suite + A0..A7 + main logs
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
with the matching classifier row. A learnable residual, initialized to 0.05,
mixes this pyramid prediction with the previous relation-weighted logit fusion.
The established 3 x 3 logit path is therefore the stable 95% anchor at
initialization; the feature pyramid is a small refinement rather than a
replacement by a uniform scale average.

An online target-prior estimator uses only raw independent-scale logits. It
maintains a soft source confusion matrix for every scale and matches all target
scale mean probabilities jointly with a constrained ridge label-shift solve.
The estimated target prior is floored and normalized. In the current method its
correction strength defaults to zero: the estimate is diagnostic-only and
cannot change final logits, pseudo-labels, domain alignment, or prototypes.
This isolation is intentional because direct label-shift inversion was unstable
under cross-dataset conditional shift. A nonzero correction remains an explicit
experimental option rather than part of A6/B6 defaults.

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

New fixed-final A6/B6 runs use isolated paths:

```text
results_boundary_reliable_pyramid/A6/
results_boundary_reliable_pyramid/B6/
logs/boundary_reliable_pyramid/A6.log
logs/boundary_reliable_pyramid/B6.log
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

The final prediction is the argmax of the relation-weighted logits plus the
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
- final predictions use bounded source-excess and boundary-attractor
  suppression but no target-prior correction;
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

For the controlled A6/B6 comparison, keep the experiment tags unchanged and
change only the evaluation protocol:

```text
results_caga_target_selected/A6/
results_caga_target_selected/B6/
logs/caga_target_selected/A6.log
logs/caga_target_selected/B6.log
```

The formal experiment matrix is:

| Tag | Purpose |
|---|---|
| A0 / A1 / A2 | 1 s / 2 s / 4 s single-scale baselines with the same class-conditional framework |
| A3 | naive multiscale attention with fused-only CDAN and no prototype reliability |
| A4 | multiscale attention with global per-scale alignment and no class prototypes |
| A5 | class-conditional alignment and prototypes, but uniform 1/2/4 s fusion |
| A6 / A7 | full multiscale method with only SEED-VII / only SEED-IV as source |
| main | full two-source, three-scale, class-conditional method |
| B6 | A6 architecture in the reverse SEED-V to SEED-VII direction |

On CSU, start the complete dual-GPU suite from the `MPUS_GA` directory:

```bash
bash scripts/run_main_A0_A7_suite.sh
```

The controller detaches itself with `nohup`, then runs two independent queues:
physical GPU 0 receives `A0 A2 A4 A6 main`, and physical GPU 1 receives
`A1 A3 A5 A7`. Each GPU starts its next experiment immediately after its own
current experiment exits; there is no GPU-idle polling. It creates exactly nine
experiment result directories and ten fixed logs under
`logs/class_conditional_multiscale/`: `suite.log`, `A0.log` through `A7.log`,
and `main.log`. A new launch overwrites these ten logs. It writes no PID,
status, or JSONL side files. Training traces are embedded in each fold's result
JSON.

Override the default fold set when doing a smoke run, for example:

```bash
TARGET_SUBJECTS=1 RANDOM_SEEDS="42" \
  bash scripts/run_main_A0_A7_suite.sh
```

Results are written under
`results_class_conditional_multiscale/{A0,A1,A2,A3,A4,A5,A6,A7,main}/`.
Completed folds are skipped by default; set `OVERWRITE=1` on the controller to
replace them.

B6 is intentionally not part of the dual-GPU suite. It uses the same model and
optimization settings as A6, but runs 20 SEED-VII target subjects for seeds 42,
43, and 44 (60 folds). Source normalization is fitted on SEED-V only. Under the
default `fixed_final` protocol, target labels remain unavailable until the
single final evaluation; under `caga_target_best`, they are accessed at the
configured evaluation interval for target-Accuracy model selection. Use a
separate `--result-root` and log path for each bidirectional comparison run so
historical `results_seedv2vii/B6/` outputs are not overwritten. The current
independent-scale/warmup verification run stores A6 and B6 together under
`results_independent_scale_warmup_fix/{A6,B6}/` and its logs under
`logs/independent_scale_warmup_fix/`.

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
