import os
import numpy as np


RANDOM_SEED = int(os.environ.get("PACKING_RANDOM_SEED", "42"))


def run_packing(num_circles):
    """
    Simple baseline: uniform grid on a 1×1 square rectangle.
    Does not optimise the aspect ratio — width = height = 1.0.
    All circles share the same radius (80% of the half-cell-size).
    """
    n = num_circles  # 21
    width, height = 1.0, 1.0

    cols = int(np.ceil(np.sqrt(n)))   # 5
    rows = int(np.ceil(n / cols))     # 5

    cell_w = width  / cols
    cell_h = height / rows

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

    # 80% of the half-cell-size → circles stay well within their cells
    radius = min(cell_w, cell_h) / 2 * 0.80
    radii  = np.full(n, radius)

    return centers, radii, float(width), float(height)


if __name__ == "__main__":
    c, r, w, h = run_packing(21)
    print(f"width={w:.4f}  height={h:.4f}  sum_radii={np.sum(r):.4f}")
