# CAGA-SGA
This repository contains the CAGA-SGA implementation for cross-dataset EEG emotion recognition.


## 1. Overview

This codebase is designed for cross-dataset EEG emotion recognition. The training script uses source-domain EEG samples for supervised learning and target-domain EEG samples for adaptation and evaluation.

The main workflow is:

```text
prepare dataset
load one target-subject fold
train with source-domain and target-domain batches
evaluate on the target-domain test split
save result of each target subject
```

---

## 2. Repository Structure

```text
CAGA-SGA/
├── README.md
├── main.py              # Main training and evaluation script
├── datapipe.py          # SEED-IV/SEED-VII -> SEED-V loading and normalization
├── model.py             # Main network implementation
├── layers.py            # Graph stream, attention layers, GRL, domain classifier
├── graph_align.py       # Semantic graph alignment module
└── golden_style.py      # Golden style bank and style alignment module
```

---

## 3. Environment

Create a conda environment:

```bash
conda create -n caga python=3.10 -y
conda activate caga

```

Install the required packages according to the imports in the code:

```bash
pip install numpy pandas scipy scikit-learn openpyxl
pip install torch torchvision torchaudio
pip install torch_geometric
```

If `torch_geometric` reports CUDA-related errors, install the PyG version that matches your local PyTorch and CUDA versions.

The main imported packages are:

```text
numpy
pandas
scipy
scikit-learn
openpyxl
torch
torch_geometric
```

---

## 4. Dataset

This repository is intended for EEG emotion recognition experiments on the SEED series datasets.

Official SEED dataset website:

```text
https://bcmi.sjtu.edu.cn/home/seed/
```

The related datasets include:

| Dataset | Emotion categories |
|---|---|
| SEED | positive, negative, neutral |
| SEED-IV | happy, sad, fear, neutral |
| SEED-V | happy, sad, fear, disgust, neutral |
| SEED-VII | happy, sad, fear, disgust, neutral, anger, surprise |


---

## 5. Run

The supported reproduction tasks are **SEED-VII -> SEED-V** and
**SEED-IV -> SEED-V**. The loader expects the following server layout:

```text
/dataset/gzw/seed_series/
├── feature/seed_iv/eeg_feature_smooth/{1,2,3}/*.mat
├── feature/seed_vii/EEG_features/{1..20}.mat
├── feature/seed_v/EEG_DE_features/{1..16}_123.npz
├── labels/SEED_IV/ReadMe.txt
└── labels/SEED_VII/emotion_label_and_stimuli_order.xlsx
```

It uses the official smoothed DE (`de_LDS`) features, maps both datasets to
positive/neutral/negative, pads every trial to 90 temporal positions, and
supports strict source-only or robust unlabeled per-domain normalization.
Temporal content is truncated at the maximum observed source length before
padding: 90 positions for SEED-VII and 64 for SEED-IV. Thus SEED-V positions
beyond 64 are excluded in the SEED-IV -> SEED-V task instead of entering the
model through source-untrained input dimensions.
Normalization statistics are fitted only on real DE windows; padded positions
remain exact zeros after standardization so trial duration is not introduced as
a cross-dataset shortcut feature.
The standardization is performed independently for every channel-time-band
feature, matching Equation (2) rather than pooling statistics across EEG
electrodes.

Validate the data without constructing or training a model:

```bash
conda activate BCI
python main.py --validate-data-only --target-subjects all
```

Validate SEED-IV -> SEED-V:

```bash
python main.py --source-dataset seed-iv --validate-data-only --target-subjects all
```

Run the complete 16-subject experiment:

```bash
conda activate BCI
python main.py --device cuda:0 --target-subjects all
```

Paper reference values for the two supported transfers are:

| Transfer | Accuracy | Macro-F1 |
|---|---:|---:|
| SEED-VII -> SEED-V | 61.73 ± 3.93% | 62.14 ± 3.79% |
| SEED-IV -> SEED-V | 70.51 ± 2.19% | 70.79 ± 2.07% |

Useful arguments include `--target-subjects 1`, `--batch-size`, `--data-root`,
and `--result-dir`. Subject numbers exposed by the CLI are one-based. The
training entry point is intentionally locked to SEED-VII -> SEED-V and 1000
iterations; SEED-IV remains available to `--validate-data-only` but is not part
of this fixed comparison protocol.

The three-class mapping is imbalanced: SEED-VII has a 30%/10%/60% split,
SEED-IV has a 25%/25%/50% split, and each SEED-V subject has a 20%/20%/60%
split. The default training command therefore uses three label-free
anti-collapse safeguards:

- source examples are sampled uniformly by class;
- target pseudo-labels enter semantic graph alignment only when all three
  predicted classes clear the confidence threshold, with equal counts per
  class;
- each target subject is normalized using only its own unlabeled feature
  statistics, compensating for the incompatible numerical scales of the two
  official pre-extracted DE feature packages on the server.

For an ablation with the uncorrected behavior, use
`--source-sampling natural --pseudo-selection threshold`. These two choices are
implementation safeguards necessitated by the missing data pipeline and are
reported separately from the paper's Table I hyperparameters.

The default `--target-normalization domain` is the robust setting verified for
the available server data. It fits target normalization statistics using only
the unlabeled trials of the current target subject. Use
`--target-normalization source` for the strict Equation (2) protocol. The
domain setting is a transparent implementation correction for the supplied
pre-extracted files, not the paper's stated normalization protocol. The result
CSV records the selected sampling, pseudo-label, and normalization modes.

Both this CAGA-SGA baseline and `MPUS_GA` Trial Temporal share the protocol in
`MPUS_GA/protocols/fixed_transductive_uda.py`:

- all 20 SEED-VII source subjects and all 1600 labeled source trials train the
  model;
- all 45 trials of the current SEED-V target subject participate in adaptation
  through a view that does not return target labels;
- training always runs for exactly 1000 iterations;
- neither target labels nor a source-domain validation split selects an epoch
  or checkpoint;
- the iteration-1000 model is evaluated exactly once on those same 45 target
  trials.

Results from this protocol are isolated from legacy outputs under:

```text
./result/transductive_fixed1000/
```

---

## 6. Hyperparameters

The reproduction hyperparameters are listed below.

| Hyperparameter | Value |
|---|---:|
| Batch Size | 48 |
| Learning Rate | $5\times10^{-4}$ |
| Weight Decay | $1\times10^{-4}$ |
| Max Iters | 1000 |
| Optimizer | AdamW |
| Dropout Rate | 0.3 |
| Feature Dimension $d_{\mathrm{model}}$ | 64 |
| CAGA Layers $L$ | 3 |
| Attention Heads | 4 |
| Top-k Sparsification | 8 |
| GCN Hidden Dimension | 64 |
| $\lambda_{\mathrm{dis}}$ | 0.1 |
| $\lambda_{\mathrm{style}}$ | 0.1 |
| $\lambda_{\mathrm{pres}}$ | 1.0 |
| $\lambda_{\mathrm{gold}}$ | 1.0 |
| $\lambda_{\mathrm{align}}$ | 1.0 |
| $\lambda_{\mathrm{gram}}$ | 0.1 |
| $\lambda_e$ | 1.0 |
| $\lambda_v$ | 1.0 |
| Pseudo-label Threshold $\delta$ | 0.90 |
| GRL Loss Cap $\tau$ | 1.0 |
| Numerical Stability $\epsilon$ | $1\times10^{-5}$ |
