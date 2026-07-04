"""
evaluate_all.py — 汇总评测脚本（禁止修改）

默认评测两个子任务的 baseline.py，也可指定其他文件：

    python evaluate_all.py                                  # 评测 baseline.py
    python evaluate_all.py solution.py                      # 评测 solution.py
    python evaluate_all.py --rect path/a.py --sq path/b.py  # 分别指定
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── 动态加载子任务评测模块 ──────────────────────────────────────────────────

def _load(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_eval_rect = _load(
    "eval_rect",
    os.path.join(_HERE, "task_A", "evaluate.py"),
)
_eval_sq = _load(
    "eval_sq",
    os.path.join(_HERE, "task_B", "evaluate.py"),
)

# ── 入口 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="圆填充汇总评测（禁止修改）")
    parser.add_argument(
        "--rect",
        default=os.path.join(_HERE, "task_A", "baseline.py"),
        help="矩形任务代码路径（默认 task_A/baseline.py）",
    )
    parser.add_argument(
        "--sq",
        default=os.path.join(_HERE, "task_B", "baseline.py"),
        help="正方形任务代码路径（默认 task_B/baseline.py）",
    )
    # 快捷用法：python evaluate_all.py solution.py
    # 将同名文件分别在两个子目录中查找
    parser.add_argument(
        "filename", nargs="?", default=None,
        help="在两个子目录中同时评测同名文件（如 baseline.py 或 solution.py）",
    )
    args = parser.parse_args()

    if args.filename:
        path_rect = os.path.join(_HERE, "task_A",  args.filename)
        path_sq   = os.path.join(_HERE, "task_B", args.filename)
    else:
        path_rect = os.path.abspath(args.rect)
        path_sq   = os.path.abspath(args.sq)

    score_a = _eval_rect.evaluate(path_rect)
    score_b = _eval_sq.evaluate(path_sq)

    combined = (score_a + score_b) / 2.0
    sep = "=" * 60
    print(f"{sep}")
    print(f"  Final Score")
    print(sep)
    print(f"  task_A (Circle Packing in Rectangle)   :  {score_a:.6f}")
    print(f"  task_B (Circle Packing in Unit Square) :  {score_b:.6f}")
    print(f"  Combined                               :  {combined:.6f}")
    print(f"{sep}\n")
