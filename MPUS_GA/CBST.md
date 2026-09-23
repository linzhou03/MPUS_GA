# Prototype-free R2 + trial-level CBST

Reference: user-provided `4.md` and [official CBST repository](https://github.com/yzou2/CBST),
commit `24439a54d1a9e5f29b8347603092b4ea3b9e9071`.
Independent implementation of paper Eq. 8 / Algorithm 2 at EEG trial level.

## One method, two requested reporting protocols

| Server | Run | Reporting |
| --- | --- | --- |
| xju | cbst_xju_fixed_metrics_20260922_s43_v2 | Formal result at step 1000; metrics logged every step |
| csu | cbst_csu_best_metrics_20260922_s43_v2 | Highest target test Accuracy over steps 1..1000; earliest tie |

Both use seed 43, one main model, six directions, no ablations. Per server:
GPU 0 runs A VII→V (16 subjects), C IV→V (16), E IV→VII (20);
GPU 1 runs B V→VII (20), D V→IV (15), F VII→IV (15).
102 subject folds per server; 204 total. Switch interval is 1 second.

## Actual training changes

- Retain the frozen R2 encoder, fusion, source losses, domain loss weight 0.2,
  source/target batch 24/16, learning rate, 300-step source warmup, ramp through
  600 and fixed 1000 optimization updates. No automatic early stopping.
- Disable prototype alignment, source prototype memory, prototype anchors,
  subgroup clustering/alignment and EMA teacher. The user's locations 2/3 are
  disabled, not revived with a new threshold. Runtime assertions enforce this.
- CBST replaces the active training consensus, its label and valid mask. The
  original standalone consensus function remains usable for old experiments and
  diagnostics. No fixed 0.6/0.9 confidence, scale-vote or JSD gate is applied after
  CBST; the CBST mask also selects target samples for the retained domain loss.
- No kNN, historical probability averaging or entropy-weighted CE in this run.

Each round collects **every unique unlabeled target trial for the target subject**
once using the current model in eval mode. Raw probabilities are the mean of the
three calibrated scale probabilities. For each raw predicted class c, sort its
confidence descending and choose threshold tau_c at rank floor(p*N_c).
For small EEG groups use max(1, floor(p*N_c)); this explicitly differs from the
official pixel implementation's zero-rank fallback, to avoid excluding a present
class just because it has fewer than 5 trials. Absent classes have tau_c=1.

Compute s_c=p_c/tau_c, label=argmax(s), accepted=max(s)>=1. Threshold ties are
included, matching the official code. These scores are not calibrated confidence.
The requested fraction is a class-relative reference, not an exact final count:
reassignment and ties can change the counts. An entirely absent raw predicted
class is not fabricated by a forced class quota.

Refresh before step 301, then every 50 updates (351,401,...,951). p starts at 20%,
increases 5 percentage points per refresh, and caps at 50% from step 601.
Freeze labels and acceptance by trial identity between refreshes, as alternating
self-training does. Do not recursively feed adjusted probabilities into a bank.

Target loss is **mean CE over accepted trials**, zero with zero gradients if a
batch accepts none. Coefficient is 1 times the existing adaptation ramp. The
threshold reward term in the paper is constant w.r.t. model weights during a
fixed-label round and does not need an extra backpropagated term.

No activation checkpointing, window chunking or allocator memory cap is enabled
in this initial configuration. Models, batch sizes and training configuration
are identical between the two servers; target evaluation preserves RNG/modes.

## Reporting and files

The csu score uses target test labels for checkpoint selection and is explicitly
marked `test_best`, not a held-out UDA score. Labels never select CBST thresholds,
modify the pseudo-label cache, update gradients or change the loss schedule.
All metrics refer to one checkpoint per subject, chosen by Accuracy.

Both servers write one parseable `TARGET_METRICS` JSON line after every update:
`current_acc`, `current_recall` for positive/neutral/negative, `best_iteration`,
`best_acc`, and `best_recall`. `best_recall` is the recall vector from the same
checkpoint that produced `best_acc`; recalls are never maximized independently.
On xju the historical best fields are diagnostics only and formal reporting stays
at update 1000.

Release: `/home/gzw/projects/MPUS_GA_releases/cbst_metrics_20260922_s43_v2/MPUS_GA`.
Environments: xju `/home/gzw/anaconda3/envs/BCI/bin/python`;
csu `/home/gzw/miniforge3/envs/BCI/bin/python`. No new environment.

Under `/home/gzw/projects/MPUS_GA`:
- `logs/<run>/suite.log` and `logs/<run>/main/seed_43/{A,...,F}.log`;
- `results_<run>/<direction>/seed_43_subject_XX.json`: formal metrics/trace;
- `.pt`: actual selected model/context/CBST cache; `.pseudo.pt`: full round and
  training-batch diagnostic data, raw vs class-relative labels and acceptance;
- csu `.last.pt` and `.best.pseudo.pt`: final model and best predictions;
- `.offline.json`: truth joined after training to compute per-class accepted
  precision/correct coverage and both selected/final-step metrics.

```bash
# Run on the corresponding server:
bash /home/gzw/projects/MPUS_GA/scripts/cbst_xju_commands.sh status
bash /home/gzw/projects/MPUS_GA/scripts/cbst_csu_commands.sh status
```

Commands also support `plan`, `start`, `report`, `stop`. Suite locks prevent
duplicate workers; code/config/data manifests prevent incompatible resumption.
