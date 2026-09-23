# R3 balanced CBST ablation on xju

Run: `r3_balanced_xju_post300bal_s43_20260923_v1`.
Release: `/home/gzw/projects/MPUS_GA_releases/r3_balanced_20260923_v1/MPUS_GA`.
Environment: `/home/gzw/anaconda3/envs/BCI/bin/python`.

The study runs all six directions, all target subjects, one seed (43), 1,000
updates per fold, and evaluates every update. Each fold selects and **saves the
actual model** with the highest target-test Balanced Accuracy among updates
301–1000; equal scores retain the earlier update. This uses labeled target
test data for checkpoint selection and is a target-selected diagnostic, not a
label-blind UDA score. Target truth does not enter pseudo-label generation or
training gradients.

Two paired variants share the same data, source sampling (`alpha=1.0`), model,
optimizer, warmup and evaluation protocol:

- `strict_quota`: existing minimum-class quota and sample-mean target CE.
- `independent`: top `max(1,floor(portion * support_c))` raw predictions per
  present class; target CE averages each present class mean equally. An absent
  class contributes no target CE and does not veto other classes.

There are 12 direction/variant jobs and 204 subject folds. GPU 0 relays
A/C/E; GPU 1 relays B/D/F, with each direction's strict and independent runs
adjacent in its queue. Other users' GPU jobs are left intact.

Output structure:

```text
/home/gzw/projects/MPUS_GA/
  results_<run>/suite_manifest.json, suite_state.json, primary_summary.json
  results_<run>_strict_quota/A/seed_43_subject_01.json, .pt, .last.pt, .pseudo.pt, .best.pseudo.pt, .offline.json
  results_<run>_independent/A/seed_43_subject_01.json, ...
  logs/<run>/suite.log
  logs/<run>_strict_quota/main/seed_43/A.log, ...
  logs/<run>_independent/main/seed_43/A.log, ...
```

The selected checkpoint is `.pt`; the final update is `.last.pt`. Per-step
metrics and confusion matrices remain in `target_evaluation_trace`.

To inspect the running job on xju:

```bash
bash /home/gzw/projects/MPUS_GA_releases/r3_balanced_20260923_v1/MPUS_GA/scripts/cbst_balanced_xju_commands.sh status
bash /home/gzw/projects/MPUS_GA_releases/r3_balanced_20260923_v1/MPUS_GA/scripts/cbst_balanced_xju_commands.sh report
```

Validation: 8 local CBST tests passed; 11 CBST core/integration tests passed
in xju BCI. Real-data isolated probes passed for both variants and confirmed
that the selected model weights match the highest eligible Balanced Accuracy.
