"""Regenerate demo data and the static visualization page."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "submission" / "demo"
DATA_PATH = DEMO_DIR / "demo_data.json"
INDEX_PATH = DEMO_DIR / "index.html"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_demo_data.py"


def main() -> int:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    (DEMO_DIR / "assets").mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, str(BUILD_SCRIPT)], cwd=REPO_ROOT, check=True)
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    INDEX_PATH.write_text(render_index(data), encoding="utf-8")
    print(f"Wrote {INDEX_PATH.relative_to(REPO_ROOT)}")
    return 0


def render_index(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GeoOpt Agent 自动化优化工作台</title>
  <link rel="stylesheet" href="assets/styles.css">
</head>
<body>
  <div class="page-shell">
    <header class="topbar">
      <div class="topbar-inner">
        <div class="brand">
          <span class="brand-mark">G</span>
          <span>GeoOpt Agent</span>
        </div>
        <nav class="nav-links" aria-label="页面导航">
          <a href="#architecture">Agent 架构</a>
          <a href="#replay">优化路径</a>
          <a href="#results">最优结果</a>
          <a href="#analysis">策略分析</a>
          <a href="#delivery">复现交付</a>
        </nav>
      </div>
    </header>

    <main>
      <section class="section hero" id="home">
        <div>
          <span class="eyebrow">Hackathon 静态展示页 · 数据生成时间 <span id="generated-at">N/A</span></span>
          <h1>GeoOpt Agent</h1>
          <p class="hero-subtitle">面向圆填充问题的自动化优化工作台</p>
          <p class="hero-copy">读取任务描述、生成 solution.py、调用 evaluate.py、基于反馈迭代优化。</p>
          <div class="composer" aria-label="Agent 任务输入">
            <div class="composer-row">
              <div class="composer-input" role="textbox">读取 Task A/B 描述，生成可提交 solution.py，并基于 evaluate.py 自动优化</div>
              <button class="primary-btn" id="start-replay" type="button">启动回放</button>
            </div>
            <div class="tag-cloud" aria-label="Agent 模块标签">
              <span class="tag">Problem Parser</span>
              <span class="tag">Coder</span>
              <span class="tag">SLSQP</span>
              <span class="tag">Evaluator</span>
              <span class="tag">Archive</span>
              <span class="tag">Report</span>
            </div>
          </div>
          <div class="kpi-grid" id="kpi-grid" aria-label="核心指标"></div>
        </div>
        <div class="hero-lab" aria-hidden="true">
          <div class="lab-panel">
            <div class="lab-grid"></div>
            <div class="molecule-map">
              <span class="link link-1"></span>
              <span class="link link-2"></span>
              <span class="link link-3"></span>
              <span class="map-node node-a"></span>
              <span class="map-node node-b"></span>
              <span class="map-node node-c"></span>
              <span class="map-node node-d"></span>
            </div>
            <span class="floating-tag float-1">Problem Parser</span>
            <span class="floating-tag float-2">SLSQP</span>
            <span class="floating-tag float-3">Evaluator</span>
            <span class="floating-tag float-4">Archive</span>
          </div>
        </div>
      </section>

      <section class="section" id="architecture">
        <div class="section-heading">
          <h2>Agent 架构</h2>
          <p>从任务解析到候选归档，每个模块只负责一个清晰环节。</p>
        </div>
        <div class="architecture-grid">
          <article class="module-card">
            <span class="module-index">1</span>
            <h3>Orchestrator</h3>
            <p>总控调度，决定本轮策略、调用工具并推进优化循环。</p>
          </article>
          <article class="module-card">
            <span class="module-index">2</span>
            <h3>Coder</h3>
            <p>生成 / 修改 solution.py，并保持可提交的独立 Python 模块。</p>
          </article>
          <article class="module-card">
            <span class="module-index">3</span>
            <h3>Numeric Optimizer</h3>
            <p>执行 SLSQP、扰动与修复，持续提高圆半径和。</p>
          </article>
          <article class="module-card">
            <span class="module-index">4</span>
            <h3>Evaluator</h3>
            <p>调用官方 evaluate.py，将合法性与 score 作为反馈信号。</p>
          </article>
          <article class="module-card">
            <span class="module-index">5</span>
            <h3>Archive</h3>
            <p>保存候选、日志与最优解，支撑可追溯的回放。</p>
          </article>
        </div>
        <p class="architecture-note card">系统采用 manager + specialist tools 架构。多智能体不是多人对话，而是职责清晰的工具化协作。</p>
      </section>

      <section class="section" id="replay">
        <div class="section-heading">
          <h2>优化路径回放</h2>
          <p>从 lineage 与 trajectory 中重建候选生成、验证、归档和最优更新过程。</p>
        </div>
        <div class="segmented" aria-label="选择任务">
          <button class="active" type="button" data-timeline-task="A">Task A</button>
          <button type="button" data-timeline-task="B">Task B</button>
        </div>
        <div class="replay-layout">
          <section class="replay-panel trajectory-panel" id="trajectory-panel" aria-label="优化路径">
            <div class="panel-heading">
              <div>
                <span>Path Timeline</span>
                <h3>Optimization Trajectory</h3>
              </div>
              <div class="timeline-meta" id="timeline-meta"></div>
            </div>
            <div class="timeline" id="timeline" aria-label="最优候选时间线"></div>
          </section>
          <aside class="replay-panel replay-detail" id="timeline-detail" aria-label="候选详情"></aside>
        </div>
      </section>

      <section class="section" id="results">
        <div class="section-heading">
          <h2>最优结果</h2>
          <p>最终圆心与半径来自当前 task_A/solution.py 和 task_B/solution.py。</p>
        </div>
        <div class="results-grid">
          <article class="result-card">
            <h3>Task A 圆填充图</h3>
            <div id="result-a"></div>
          </article>
          <article class="result-card">
            <h3>Task B 圆填充图</h3>
            <div id="result-b"></div>
          </article>
        </div>
      </section>

      <section class="section" id="analysis">
        <div class="section-heading">
          <h2>策略分析</h2>
          <p>Agent 会根据评测反馈在探索、利用和修复之间切换。</p>
        </div>
        <div class="analysis-grid">
          <article class="analysis-panel">
            <h3>score trajectory</h3>
            <div id="score-chart"></div>
          </article>
          <div class="analysis-panel">
            <h3>strategy stats</h3>
            <div class="bar-list" id="strategy-bars"></div>
            <h3 style="margin-top:24px">failure stats</h3>
            <div class="bar-list" id="failure-bars"></div>
          </div>
        </div>
        <p class="analysis-note">当 Evaluator 返回低分、平台期或几何风险时，Orchestrator 会在结构化初始化、multi-start SLSQP 与 perturb-and-repair 之间切换。</p>
      </section>

      <section class="section" id="delivery">
        <div class="section-heading">
          <h2>复现与交付</h2>
          <p>以下命令与目录结构用于快速复现实验和定位提交文件。</p>
        </div>
        <div class="delivery-grid">
          <article class="code-panel">
            <h3>复现命令</h3>
            <div class="command-list">
              <pre class="command"><code>python agent/run.py --task both --iterations 5 --fast</code></pre>
              <pre class="command"><code>python task_A/evaluate.py</code></pre>
              <pre class="command"><code>python task_B/evaluate.py</code></pre>
              <pre class="command"><code>python evaluate_all.py</code></pre>
            </div>
          </article>
          <article class="code-panel">
            <h3>submission 目录结构</h3>
            <pre class="tree" id="submission-tree"></pre>
          </article>
        </div>
      </section>
    </main>

    <footer class="footer">
      本页面由 final solution.py 与 Agent 日志自动生成，不依赖网络，适合作为评委快速理解系统的入口。
    </footer>
  </div>
  <script id="demo-data" type="application/json">{payload}</script>
  <script src="assets/app.js" defer></script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
