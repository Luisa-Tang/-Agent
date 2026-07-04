# Task: Circle Packing in a Rectangle

## Overview

Place 21 non-overlapping disks inside a rectangle of perimeter 4, maximizing the sum of all radii. The rectangle's aspect ratio is a free variable that must also be optimized.

## Description

Pack `num_circles = 21` disjoint disks into a rectangle with `width + height = 2` (`width > 0`, `height > 0`), maximizing the total sum of radii.

**Constraints:**

- **Perimeter**: `width + height = 2`
- **Containment**: Each circle must lie fully inside `[0, width] × [0, height]`:
  - `x_i - r_i >= 0`, `x_i + r_i <= width`
  - `y_i - r_i >= 0`, `y_i + r_i <= height`
- **Disjointness**: No two circles overlap:
  - `sqrt((x_i - x_j)^2 + (y_i - y_j)^2) >= r_i + r_j` for all `i != j`
- **Non-negativity**: `r_i >= 0` for all `i`

**Objective**: Maximize `sum(r_i)`.

## Required Items

Implement the following function in `solution.py`:

```python
def run_packing(num_circles: int):
    # num_circles = 21
    ...
    return centers, radii, width, height
```

| Return Value | Type | Description |
|---|---|---|
| `centers` | `np.ndarray`, shape `(21, 2)` | Center coordinates `(x, y)` of each circle |
| `radii` | `np.ndarray`, shape `(21,)` | Radius of each circle; non-negative finite values |
| `width` | `float` | Rectangle width; must satisfy `2 * (width + height) == 4.0` |
| `height` | `float` | Rectangle height |
