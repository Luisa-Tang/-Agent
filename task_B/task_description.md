# Task: Circle Packing in a Unit Square

## Overview

Place 26 non-overlapping disks inside the unit square `[0, 1] × [0, 1]`, maximizing the sum of all radii. Circles may have different radii.

## Description

Pack `num_circles = 26` disjoint disks into the unit square `[0, 1] × [0, 1]`, maximizing the total sum of radii.

**Constraints:**

- **Containment**: Each circle must lie fully inside `[0, 1] × [0, 1]`:
  - `r_i <= min(x_i, 1 - x_i, y_i, 1 - y_i)`
- **Disjointness**: No two circles overlap:
  - `sqrt((x_i - x_j)^2 + (y_i - y_j)^2) >= r_i + r_j` for all `i != j`
- **Non-negativity**: `r_i >= 0` for all `i`

**Objective**: Maximize `sum(r_i)`.

## Required Items

Implement the following function in `solution.py`:

```python
def run_packing(num_circles: int):
    # num_circles = 26
    ...
    return centers, radii, sum_radii
```

| Return Value | Type | Description |
|---|---|---|
| `centers` | `np.ndarray`, shape `(26, 2)` | Center coordinates `(x, y)` of each circle; all values in `[0, 1]` |
| `radii` | `np.ndarray`, shape `(26,)` | Radius of each circle; non-negative finite values |
| `sum_radii` | `float` | Sum of all radii; must equal `np.sum(radii)` |
