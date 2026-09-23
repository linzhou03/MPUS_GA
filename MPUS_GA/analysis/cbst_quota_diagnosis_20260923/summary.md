# Strict-quota CBST diagnosis (xju, seed 43)

Read-only analysis of all 102 completed folds in
`results_cbst_quota_xju_fixed_20260922_s43_v1`, using each fold's saved
`training_batches`, CBST refresh rounds, and `.offline.json`. The script is
[`analyze.py`](analyze.py). Target truth is joined only for this retrospective
diagnosis; it did not change training or model selection.

| Direction | Folds | Empty target batches, steps 301–1000 | Refreshes with zero accepted | Accepted trials per refresh | Accepted pseudo-label precision P / N / Neg |
|---|---:|---:|---:|---:|---:|
| A VII→V | 16 | 10.79% | 3.57% | 10.59 | 43.74 / 66.25 / 85.34% |
| B V→VII | 20 | 5.41% | 0% | 19.35 | 49.17 / 28.13 / 77.24% |
| C IV→V | 16 | 18.03% | 5.80% | 7.50 | 48.75 / 15.89 / 59.46% |
| D V→IV | 15 | 4.71% | 0% | 19.24 | 47.36 / 27.99 / 57.98% |
| E IV→VII | 20 | 13.56% | 2.14% | 17.83 | 42.19 / 0 / 63.58% |
| F VII→IV | 15 | 10.06% | 0.48% | 16.97 | 44.70 / 16.41 / 51.35% |

The precision denominators pool class-labeled accepted observations across
CBST refreshes and subjects. The same trial may recur in multiple refreshes;
these counts are descriptive and are not independent trial sample sizes. E's
Neutral precision is zero across 1,664 such accepted observations. This is a
particularly important risk for the independent-reception variant: increasing
coverage may increase wrong supervision unless pseudo-label reliability also
improves. The current paired experiment tests that consequence rather than
assuming more reception is beneficial.
