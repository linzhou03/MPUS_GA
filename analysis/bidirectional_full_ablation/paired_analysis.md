# Paired bidirectional ablation analysis

Result root: `/home/gzw/projects/MPUS_GA/results_bidirectional_full_ablation`

The target subject is the statistical unit. The three random seeds are averaged within each subject before paired differences are calculated. All results were validated as fixed-final 1000-iteration runs with one target evaluation and no checkpoint selection.

Confidence intervals are percentile paired-bootstrap intervals (20,000 resamples, seed 20260822). Wilcoxon tests are two-sided. Holm correction is applied across all planned comparisons separately for each metric. Performance changes are oriented so that positive is always better; for recall gap it means a reduction. Prediction rates in the CSV are descriptive raw candidate-minus-reference changes.

## Direction A: SEED-VII to SEED-V

| Comparison | Metric | Improvement pp [95% CI] | p | Holm p | dz | Rank-biserial |
|---|---|---:|---:|---:|---:|---:|
| A3 − A0 (uniform multiscale vs 1 s) | Accuracy | +5.32 [+2.45, +8.29] | 0.004816 | 0.06261 | 0.871 | 0.801 |
| A3 − A0 (uniform multiscale vs 1 s) | Balanced Accuracy | +0.90 [-2.03, +3.88] | 0.5098 | 1 | 0.144 | 0.200 |
| A3 − A0 (uniform multiscale vs 1 s) | Macro-F1 | +3.59 [+0.23, +6.84] | 0.02139 | 0.2353 | 0.514 | 0.647 |
| A3 − A0 (uniform multiscale vs 1 s) | Worst-class recall | +4.63 [-2.47, +10.96] | 0.14 | 1 | 0.324 | 0.457 |
| A3 − A0 (uniform multiscale vs 1 s) | Recall gap | +7.87 [-0.31, +15.43] | 0.09344 | 0.7476 | 0.469 | 0.471 |
| A3 − A1 (uniform multiscale vs 2 s) | Accuracy | +6.57 [+4.31, +8.89] | 0.0007733 | 0.0116 | 1.353 | 0.941 |
| A3 − A1 (uniform multiscale vs 2 s) | Balanced Accuracy | +1.75 [-0.39, +3.60] | 0.0928 | 1 | 0.414 | 0.485 |
| A3 − A1 (uniform multiscale vs 2 s) | Macro-F1 | +4.67 [+2.15, +6.81] | 0.006287 | 0.08801 | 0.944 | 0.750 |
| A3 − A1 (uniform multiscale vs 2 s) | Worst-class recall | +6.71 [+1.70, +11.42] | 0.02469 | 0.3457 | 0.646 | 0.675 |
| A3 − A1 (uniform multiscale vs 2 s) | Recall gap | +9.10 [+1.62, +15.97] | 0.03185 | 0.3504 | 0.608 | 0.618 |
| A3 − A2 (uniform multiscale vs 4 s) | Accuracy | +5.23 [+2.50, +7.96] | 0.004101 | 0.05741 | 0.903 | 0.809 |
| A3 − A2 (uniform multiscale vs 4 s) | Balanced Accuracy | +3.58 [+0.82, +6.22] | 0.02982 | 0.4175 | 0.627 | 0.610 |
| A3 − A2 (uniform multiscale vs 4 s) | Macro-F1 | +5.22 [+1.76, +8.31] | 0.007629 | 0.09918 | 0.761 | 0.735 |
| A3 − A2 (uniform multiscale vs 4 s) | Worst-class recall | +9.03 [+3.63, +13.35] | 0.009713 | 0.1554 | 0.872 | 0.735 |
| A3 − A2 (uniform multiscale vs 4 s) | Recall gap | +11.57 [+4.24, +17.98] | 0.01995 | 0.2594 | 0.801 | 0.662 |
| A_main − A3 (full class-conditional method vs uniform multiscale) | Accuracy | +0.65 [-0.88, +2.13] | 0.4222 | 1 | 0.203 | 0.243 |
| A_main − A3 (full class-conditional method vs uniform multiscale) | Balanced Accuracy | +0.41 [-1.26, +2.34] | 0.8563 | 1 | 0.108 | -0.051 |
| A_main − A3 (full class-conditional method vs uniform multiscale) | Macro-F1 | +0.37 [-1.44, +2.36] | 0.9399 | 1 | 0.091 | 0.029 |
| A_main − A3 (full class-conditional method vs uniform multiscale) | Worst-class recall | -1.08 [-4.24, +2.24] | 0.6047 | 1 | -0.156 | -0.206 |
| A_main − A3 (full class-conditional method vs uniform multiscale) | Recall gap | -2.08 [-7.02, +2.78] | 0.4225 | 1 | -0.199 | -0.199 |
| A_main − A4 (feature-pyramid contribution) | Accuracy | +0.83 [-1.16, +2.82] | 0.4098 | 1 | 0.197 | 0.233 |
| A_main − A4 (feature-pyramid contribution) | Balanced Accuracy | +0.21 [-2.03, +2.37] | 0.7173 | 1 | 0.044 | 0.088 |
| A_main − A4 (feature-pyramid contribution) | Macro-F1 | +0.56 [-2.25, +2.96] | 0.4332 | 1 | 0.103 | 0.235 |
| A_main − A4 (feature-pyramid contribution) | Worst-class recall | -0.54 [-5.94, +4.24] | 0.9499 | 1 | -0.050 | 0.000 |
| A_main − A4 (feature-pyramid contribution) | Recall gap | -0.69 [-6.40, +4.71] | 0.7762 | 1 | -0.059 | -0.092 |
| A_main − A5 (boundary-attractor contribution) | Accuracy | +0.42 [-0.51, +1.39] | 0.4073 | 1 | 0.208 | 0.221 |
| A_main − A5 (boundary-attractor contribution) | Balanced Accuracy | +1.05 [-0.28, +2.49] | 0.2219 | 1 | 0.362 | 0.381 |
| A_main − A5 (boundary-attractor contribution) | Macro-F1 | +0.92 [-0.42, +2.39] | 0.3755 | 1 | 0.310 | 0.265 |
| A_main − A5 (boundary-attractor contribution) | Worst-class recall | +3.32 [+0.15, +6.71] | 0.1312 | 1 | 0.481 | 0.467 |
| A_main − A5 (boundary-attractor contribution) | Recall gap | +3.09 [-1.16, +7.33] | 0.1572 | 0.9814 | 0.344 | 0.400 |
| A5 − A6 (source-excess contribution with boundary disabled) | Accuracy | +4.21 [+1.53, +7.27] | 0.01091 | 0.1091 | 0.691 | 0.752 |
| A5 − A6 (source-excess contribution with boundary disabled) | Balanced Accuracy | -0.90 [-2.31, +0.87] | 0.08292 | 0.9951 | -0.265 | -0.485 |
| A5 − A6 (source-excess contribution with boundary disabled) | Macro-F1 | +1.79 [-0.29, +4.23] | 0.1928 | 1 | 0.376 | 0.382 |
| A5 − A6 (source-excess contribution with boundary disabled) | Worst-class recall | +0.31 [-2.08, +2.93] | 0.8609 | 1 | 0.058 | 0.066 |
| A5 − A6 (source-excess contribution with boundary disabled) | Recall gap | +4.86 [+1.54, +8.18] | 0.01995 | 0.2594 | 0.695 | 0.647 |
| A_main − A6 (full method vs both suppressors disabled) | Accuracy | +4.63 [+1.81, +7.69] | 0.007122 | 0.07834 | 0.758 | 0.757 |
| A_main − A6 (full method vs both suppressors disabled) | Balanced Accuracy | +0.15 [-1.47, +1.88] | 0.8752 | 1 | 0.043 | 0.029 |
| A_main − A6 (full method vs both suppressors disabled) | Macro-F1 | +2.70 [+0.32, +4.99] | 0.01825 | 0.219 | 0.554 | 0.662 |
| A_main − A6 (full method vs both suppressors disabled) | Worst-class recall | +3.63 [-0.62, +8.10] | 0.14 | 1 | 0.391 | 0.457 |
| A_main − A6 (full method vs both suppressors disabled) | Recall gap | +7.95 [+2.70, +13.50] | 0.01504 | 0.2383 | 0.696 | 0.700 |

## Direction B: SEED-V to SEED-VII

| Comparison | Metric | Improvement pp [95% CI] | p | Holm p | dz | Rank-biserial |
|---|---|---:|---:|---:|---:|---:|
| B3 − B0 (uniform multiscale vs 1 s) | Accuracy | +4.75 [+3.54, +5.94] | 0.0001954 | 0.003127 | 1.718 | 1.000 |
| B3 − B0 (uniform multiscale vs 1 s) | Balanced Accuracy | +2.07 [+0.61, +3.61] | 0.01758 | 0.2637 | 0.581 | 0.605 |
| B3 − B0 (uniform multiscale vs 1 s) | Macro-F1 | +3.07 [+1.97, +4.19] | 2.67e-05 | 0.0004272 | 1.190 | 0.943 |
| B3 − B0 (uniform multiscale vs 1 s) | Worst-class recall | +1.18 [-1.42, +3.72] | 0.3045 | 1 | 0.194 | 0.289 |
| B3 − B0 (uniform multiscale vs 1 s) | Recall gap | -0.24 [-4.03, +3.58] | 0.8227 | 1 | -0.027 | -0.063 |
| B3 − B1 (uniform multiscale vs 2 s) | Accuracy | +0.62 [-1.04, +2.31] | 0.6142 | 1 | 0.159 | 0.119 |
| B3 − B1 (uniform multiscale vs 2 s) | Balanced Accuracy | +2.25 [+0.01, +4.22] | 0.008002 | 0.128 | 0.457 | 0.652 |
| B3 − B1 (uniform multiscale vs 2 s) | Macro-F1 | +1.01 [-0.57, +2.54] | 0.2024 | 1 | 0.280 | 0.333 |
| B3 − B1 (uniform multiscale vs 2 s) | Worst-class recall | +1.32 [-1.32, +3.89] | 0.3719 | 1 | 0.215 | 0.240 |
| B3 − B1 (uniform multiscale vs 2 s) | Recall gap | -3.75 [-8.78, +1.22] | 0.167 | 0.9814 | -0.320 | -0.352 |
| B3 − B2 (uniform multiscale vs 4 s) | Accuracy | +3.19 [+1.40, +4.98] | 0.005581 | 0.06697 | 0.755 | 0.686 |
| B3 − B2 (uniform multiscale vs 4 s) | Balanced Accuracy | +1.54 [+0.29, +2.88] | 0.04185 | 0.5441 | 0.499 | 0.529 |
| B3 − B2 (uniform multiscale vs 4 s) | Macro-F1 | +2.80 [+1.36, +4.25] | 0.001432 | 0.02149 | 0.833 | 0.771 |
| B3 − B2 (uniform multiscale vs 4 s) | Worst-class recall | +4.76 [+1.70, +7.88] | 0.01682 | 0.2524 | 0.659 | 0.610 |
| B3 − B2 (uniform multiscale vs 4 s) | Recall gap | +7.40 [+2.01, +12.50] | 0.0149 | 0.2383 | 0.603 | 0.642 |
| B_main − B3 (full class-conditional method vs uniform multiscale) | Accuracy | -0.29 [-0.94, +0.35] | 0.3402 | 1 | -0.196 | -0.286 |
| B_main − B3 (full class-conditional method vs uniform multiscale) | Balanced Accuracy | +0.10 [-0.64, +0.82] | 0.6215 | 1 | 0.061 | 0.163 |
| B_main − B3 (full class-conditional method vs uniform multiscale) | Macro-F1 | +0.09 [-0.46, +0.64] | 0.5459 | 1 | 0.073 | 0.162 |
| B_main − B3 (full class-conditional method vs uniform multiscale) | Worst-class recall | +1.98 [+0.52, +3.51] | 0.02891 | 0.3758 | 0.570 | 0.602 |
| B_main − B3 (full class-conditional method vs uniform multiscale) | Recall gap | +2.53 [+0.14, +4.90] | 0.08238 | 0.7414 | 0.457 | 0.468 |
| B_main − B4 (feature-pyramid contribution) | Accuracy | -0.85 [-1.77, +0.15] | 0.05668 | 0.4534 | -0.383 | -0.495 |
| B_main − B4 (feature-pyramid contribution) | Balanced Accuracy | +0.49 [-1.56, +2.53] | 0.5016 | 1 | 0.102 | 0.167 |
| B_main − B4 (feature-pyramid contribution) | Macro-F1 | -0.50 [-1.56, +0.60] | 0.4749 | 1 | -0.195 | -0.190 |
| B_main − B4 (feature-pyramid contribution) | Worst-class recall | +0.28 [-1.42, +1.91] | 0.6317 | 1 | 0.071 | 0.082 |
| B_main − B4 (feature-pyramid contribution) | Recall gap | -3.75 [-8.09, +0.49] | 0.1402 | 0.9814 | -0.373 | -0.381 |
| B_main − B5 (boundary-attractor contribution) | Accuracy | -0.15 [-0.46, +0.17] | 0.5124 | 1 | -0.196 | -0.242 |
| B_main − B5 (boundary-attractor contribution) | Balanced Accuracy | +0.08 [-0.42, +0.56] | 0.5326 | 1 | 0.071 | 0.152 |
| B_main − B5 (boundary-attractor contribution) | Macro-F1 | -0.08 [-0.43, +0.23] | 0.8983 | 1 | -0.109 | -0.038 |
| B_main − B5 (boundary-attractor contribution) | Worst-class recall | +0.03 [-0.69, +0.73] | 0.6144 | 1 | 0.021 | 0.095 |
| B_main − B5 (boundary-attractor contribution) | Recall gap | -0.76 [-2.60, +0.66] | 0.7116 | 1 | -0.200 | -0.125 |
| B5 − B6 (source-excess contribution with boundary disabled) | Accuracy | +1.60 [+0.48, +2.94] | 0.04806 | 0.4325 | 0.552 | 0.532 |
| B5 − B6 (source-excess contribution with boundary disabled) | Balanced Accuracy | -0.19 [-1.89, +1.27] | 0.7021 | 1 | -0.050 | 0.089 |
| B5 − B6 (source-excess contribution with boundary disabled) | Macro-F1 | +0.97 [-0.08, +2.17] | 0.1893 | 1 | 0.361 | 0.343 |
| B5 − B6 (source-excess contribution with boundary disabled) | Worst-class recall | +1.35 [+0.07, +2.71] | 0.08808 | 0.9689 | 0.438 | 0.510 |
| B5 − B6 (source-excess contribution with boundary disabled) | Recall gap | +4.51 [+1.22, +8.06] | 0.01575 | 0.2383 | 0.559 | 0.667 |
| B_main − B6 (full method vs both suppressors disabled) | Accuracy | +1.46 [+0.33, +2.75] | 0.07954 | 0.5568 | 0.515 | 0.463 |
| B_main − B6 (full method vs both suppressors disabled) | Balanced Accuracy | -0.10 [-2.05, +1.39] | 0.3316 | 1 | -0.026 | 0.243 |
| B_main − B6 (full method vs both suppressors disabled) | Macro-F1 | +0.88 [-0.21, +2.06] | 0.2611 | 1 | 0.335 | 0.295 |
| B_main − B6 (full method vs both suppressors disabled) | Worst-class recall | +1.39 [-0.10, +2.78] | 0.06404 | 0.7685 | 0.413 | 0.479 |
| B_main − B6 (full method vs both suppressors disabled) | Recall gap | +3.75 [+0.17, +7.29] | 0.05078 | 0.5078 | 0.452 | 0.511 |
