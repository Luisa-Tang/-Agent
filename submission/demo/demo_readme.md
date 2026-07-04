# GeoOpt Agent 可视化 Demo

## 如何打开 demo

直接用浏览器打开：

```bash
submission/demo/index.html
```

页面不依赖外部 CDN 或网络。`demo_data.json` 会被内嵌到 `index.html`，因此双击打开也可以展示优化路径、最优解和策略统计。

## 如何重新生成

在项目根目录运行：

```bash
python scripts/make_demo_assets.py
```

该命令会先生成 `submission/demo/demo_data.json`，再更新 `submission/demo/index.html`。

## 如何复现评测

在项目根目录运行：

```bash
python evaluate_all.py
```

也可以分别运行：

```bash
python task_A/evaluate.py
python task_B/evaluate.py
```
