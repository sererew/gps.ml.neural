# LOFO Cross-Validation Summary Report%n%n## Overview%n%n- **Total Folds**: 2
- **Validation Method**: Leave-One-Family-Out

## Performance Summary%n%n### Neural Network Performance%n%n| Metric            | Mean ± Std           | Range                  |%n|-------------------|----------------------|------------------------|%n| Distance MAE      | 16099,488 ± 3355,414 | [12744,074, 19454,902] |
| Elevation Pos MAE | 543,058 ± 26,229 | [516,830, 569,287] |
| Elevation Neg MAE | 542,542 ± 26,342 | [516,200, 568,883] |
| Overall MAE       | 5728,362 ± 1100,948 | [4627,415, 6829,311] |

### Baseline Performance%n%n| Metric            | Mean ± Std           | Range                  |%n|-------------------|----------------------|------------------------|%n| Distance MAE      | 14491,262 ± 1694,698 | [12796,564, 16185,960] |
| Elevation Pos MAE | 3588,322 ± 230,066 | [3358,257, 3818,388] |
| Elevation Neg MAE | 3760,942 ± 201,941 | [3559,001, 3962,883] |
| Overall MAE       | 7280,176 ± 420,897 | [6859,279, 7701,072] |

## Fold Details%n%n### Fold 1: 1

- **Training Families**: 10, 11, 2, 3, 3a, 3b, 3c, 3d, 4, 5, 6, 7, 8, 9, 9a, 9b, 9c, 9d
- **NN MAE**: [12744,074, 569,287, 568,883] (overall: 4627,415)
- **Baseline MAE**: [12796,564, 3818,388, 3962,883] (overall: 6859,279)

### Fold 2: 10

- **Training Families**: 1, 11, 2, 3, 3a, 3b, 3c, 3d, 4, 5, 6, 7, 8, 9, 9a, 9b, 9c, 9d
- **NN MAE**: [19454,902, 516,830, 516,200] (overall: 6829,311)
- **Baseline MAE**: [16185,960, 3358,257, 3559,001] (overall: 7701,072)

