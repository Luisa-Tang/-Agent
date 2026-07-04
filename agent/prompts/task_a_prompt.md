Task A packs 21 non-overlapping circles into a rectangle with perimeter 4.

The solution must define:

```python
def run_packing(num_circles):
    return centers, radii, width, height
```

Constraints:

- `width + height = 2`
- all circles inside `[0, width] x [0, height]`
- no pair overlaps
- all outputs finite

The score is `sum(radii) / 2.365840`.
