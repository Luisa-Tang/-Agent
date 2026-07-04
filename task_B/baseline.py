import os
import numpy as np


RANDOM_SEED = int(os.environ.get("PACKING_RANDOM_SEED", "42"))


def run_packing(num_circles):
    """
    Simple baseline: uniform grid on the unit square.
    Uses a 6×5 grid (30 positions) and takes the first 26.
    All circles share the same radius (90% of half the shorter cell side).
    """
    n = num_circles  # 26

    cols, rows = 6, 5
    cell_w = 1.0 / cols   # ≈ 0.1667
    cell_h = 1.0 / rows   # = 0.2000

    centers = []
    for r in range(rows):
        for c in range(cols):
            if len(centers) >= n:
                break
            cx = cell_w * c + cell_w / 2
            cy = cell_h * r + cell_h / 2
            centers.append([cx, cy])
        if len(centers) >= n:
            break

    centers = np.array(centers[:n])

    # 90% of the half-cell-size of the shorter dimension
    radius = min(cell_w, cell_h) / 2 * 0.90
    radii  = np.full(n, radius)

    return centers, radii, float(np.sum(radii))


if __name__ == "__main__":
    c, r, s = run_packing(26)
    print(f"sum_radii={s:.4f}")
