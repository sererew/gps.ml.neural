# LOFO Cross-Validation Summary Report

## Overview

- **Total Folds**: 2
- **Validation Method**: Leave-One-Family-Out

## Performance Summary

### Neural Network Performance

| Metric            | Mean ± Std | Range |
|-------------------|------------|-------|
| Distance MAE      | 16099,490 ± 3355,414 | [12744,075, 19454,904] |
| Elevation Pos MAE | 543,058 ± 26,229 | [516,830, 569,287] |
| Elevation Neg MAE | 542,541 ± 26,342 | [516,200, 568,883] |
| Overall MAE       | 5728,363 ± 1100,948 | [4627,415, 6829,311] |

### Baseline Performance

| Metric            | Mean ± Std | Range |
|-------------------|------------|-------|
| Distance MAE      | 14491,236 ± 1694,685 | [12796,551, 16185,921] |
| Elevation Pos MAE | 3588,311 ± 230,066 | [3358,245, 3818,377] |
| Elevation Neg MAE | 3760,969 ± 201,933 | [3559,036, 3962,902] |
| Overall MAE       | 7280,172 ± 420,895 | [6859,277, 7701,067] |

## Fold Details

### Fold 1: 1

- **Training Families**: 10, 11, 2, 3, 3a, 3b, 3c, 3d, 4, 5, 6, 7, 8, 9, 9a, 9b, 9c, 9d
- **NN MAE**: [12744,075, 569,287, 568,883] (overall: 4627,415)
- **Baseline MAE**: [12796,551, 3818,377, 3962,902] (overall: 6859,277)

### Fold 2: 10

- **Training Families**: 1, 11, 2, 3, 3a, 3b, 3c, 3d, 4, 5, 6, 7, 8, 9, 9a, 9b, 9c, 9d
- **NN MAE**: [19454,904, 516,830, 516,200] (overall: 6829,311)
- **Baseline MAE**: [16185,921, 3358,245, 3559,036] (overall: 7701,067)

