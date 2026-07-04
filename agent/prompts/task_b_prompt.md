Task B packs 26 non-overlapping circles into the unit square.

The solution must define:

```python
def run_packing(num_circles):
    return centers, radii, sum_radii
```

Constraints:

- centers have shape `(26, 2)`
- radii have shape `(26,)`
- circles remain inside `[0, 1] x [0, 1]`
- no pair overlaps
- `sum_radii == np.sum(radii)`

The score is `sum(radii) / 2.635990`.
