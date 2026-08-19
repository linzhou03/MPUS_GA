# MPUS-GA multiscale DE project

This subproject is isolated from the CAGA-SGA reproduction code. It rebuilds
SEED-VII and SEED-V differential-entropy features directly from raw CNT EEG at
three non-overlapping window lengths: 1 s, 2 s, and 4 s.

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
├── eeg_raw/SEED_VII/*.cnt
├── eeg_raw/SEED_V/*.cnt
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
MNE_DONTWRITE_HOME=true /home/gzw/anaconda3/envs/BCI/bin/python \
  MPUS_GA/preprocess.py \
  --data-root /dataset/gzw/seed_series \
  --datasets seed-vii seed-v \
  --window-seconds 1 2 4
```

Use explicit paths if automatic discovery does not match the server:

```bash
MNE_DONTWRITE_HOME=true /home/gzw/anaconda3/envs/BCI/bin/python \
  MPUS_GA/preprocess.py \
  --seed-vii-raw-dir /path/to/SEED_VII/EEG_raw \
  --seed-v-raw-dir /path/to/SEED_V/EEG_raw \
  --seed-vii-label-file /path/to/emotion_label_and_stimuli_order.xlsx
```

For a small pipeline check:

```bash
MNE_DONTWRITE_HOME=true /home/gzw/anaconda3/envs/BCI/bin/python \
  MPUS_GA/preprocess.py --datasets seed-vii --subjects 1 --window-seconds 1 2 4
```

Validate all generated files:

```bash
/home/gzw/anaconda3/envs/BCI/bin/python MPUS_GA/validate_processed.py
```

## Transfer training

`train_transfer.py` treats every DE window as one 62-node EEG graph. Source
normalization statistics are fitted once on SEED-VII and applied to SEED-V by
default. SEED-V labels remain attached only for the final evaluation after the
fixed training budget; they are not read by normalization, sampling, teacher
prediction, pseudo-label selection, or optimization.

Stage 0 freezes the CAGA-SGA balanced baseline for the requested three scales.
SEED-VII and SEED-V are dataset names, not random seeds; optimization uses the
single fixed random seed 42 by default:

```bash
MNE_DONTWRITE_HOME=true /home/gzw/anaconda3/envs/BCI/bin/python \
  MPUS_GA/train_transfer.py \
  --variant caga-balanced \
  --window-seconds 1 2 4 \
  --random-seeds 42 \
  --target-subjects all \
  --device cuda:1
```

The first-priority MVP from the proposal adds an EMA teacher, three stochastic
teacher passes, entropy-plus-MI reliability, balanced target-node gating,
separately normalized SS/ST/TT soft edge losses, and the 200+200 iteration graph
ramp:

```bash
MNE_DONTWRITE_HOME=true /home/gzw/anaconda3/envs/BCI/bin/python \
  MPUS_GA/train_transfer.py \
  --variant r-softsga \
  --window-seconds 1 2 4 \
  --random-seeds 42 \
  --target-subjects all \
  --device cuda:1
```

The terminal displays an overall tqdm bar across all scale/subject folds, an
iteration bar with live losses for the current fold, and a final-evaluation
batch bar. With one method, three scales, one random seed, and 16 target
subjects, the overall bar contains 48 folds.

Runs are resumable at subject granularity: an existing `subject_XX.json` is
skipped unless `--overwrite` is supplied. Results contain both window-level
metrics and trial-level metrics obtained by averaging all window probabilities
within a trial. The default output is
`MPUS_GA/results/<variant>/window_*/random_seed_*`.

For a cheap end-to-end smoke test before the full matrix:

```bash
/home/gzw/anaconda3/envs/BCI/bin/python MPUS_GA/train_transfer.py \
  --variant r-softsga --window-seconds 4 --random-seeds 42 \
  --target-subjects 16 --max-iters 2 --batch-size 8 --device cuda:1
```

### Sequential nohup run

`run_transfer_sequential.sh` first verifies the expected 80 SEED-VII and 48
SEED-V session files at every scale, runs the full processed-data validator,
then executes `caga-balanced` followed by `r-softsga`. A failed validation or
baseline stops the sequence rather than launching the next stage. Completed
subject results are reused when the script is restarted.

Launch the script itself with nohup:

```bash
cd /home/gzw/projects/CAGA-SGA
mkdir -p MPUS_GA/logs/training
RUN_LOG="MPUS_GA/logs/training/nohup_$(date +%Y%m%d_%H%M%S).log"
nohup bash MPUS_GA/run_transfer_sequential.sh > "${RUN_LOG}" 2>&1 &
echo "PID=$! LOG=${RUN_LOG}"
```

Follow the master status log and the detailed tqdm logs with:

```bash
tail -f "${RUN_LOG}"
tail -f MPUS_GA/logs/training/*_caga-balanced.log
tail -f MPUS_GA/logs/training/*_r-softsga.log
```

The script accepts environment overrides without editing the file, for
example `DEVICE=cuda:0`, `MAX_ITERS=2`, or `BATCH_SIZE=32`. The checked-in
default is `cuda:1`.

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
