(function () {
  "use strict";

  const strategyNames = {
    baseline_safe_grid: "安全网格",
    benchmark_seed_dominikkamp: "基准种子",
    hexagonal_or_staggered_initialization: "交错初始化",
    scipy_slsqp_joint: "SLSQP 联合优化",
    multi_start_slsqp: "多起点 SLSQP",
    perturb_best_and_repair: "最优扰动修复",
    unknown: "未知策略",
  };

  const failureNames = {
    none: "无",
    low_score: "得分偏低",
    plateau: "平台期",
    overlap: "圆重叠",
    boundary_violation: "边界越界",
    timeout: "超时",
    shape_error: "形状错误",
    nonfinite: "非有限数值",
    negative_radius: "负半径",
    perimeter_error: "周长约束错误",
    unknown: "未知",
  };

  const colors = ["#2563eb", "#06b6d4", "#10b981", "#f59e0b", "#8b5cf6", "#ef5da8"];

  function demoData() {
    const node = document.getElementById("demo-data");
    if (!node) return window.GEOOPT_DEMO_DATA || {};
    try {
      return JSON.parse(node.textContent);
    } catch (error) {
      console.warn("demo_data 解析失败", error);
      return window.GEOOPT_DEMO_DATA || {};
    }
  }

  function fmt(value, digits) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "N/A";
    return n.toFixed(digits === undefined ? 6 : digits);
  }

  function sci(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "N/A";
    return n.toExponential(3);
  }

  function byId(id) {
    return document.getElementById(id);
  }

  function translateStrategy(strategy) {
    return strategyNames[strategy] || strategy || "未知策略";
  }

  function strategyCode(strategy) {
    if (!strategy || strategy === "unknown") return "";
    return strategy;
  }

  function translateFailure(failure) {
    return failureNames[failure] || failure || "未知";
  }

  function escapeHtml(value) {
    return String(value === undefined || value === null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function boolText(value) {
    return value ? "Valid" : "Invalid";
  }

  function statusClass(record) {
    if (!record || !record.valid) return "invalid";
    return record.failure_type && record.failure_type !== "none" ? "warning" : "valid";
  }

  function statusText(record) {
    if (!record || !record.valid) return "invalid";
    if (record.failure_type && record.failure_type !== "none") return translateFailure(record.failure_type);
    return "valid";
  }

  function translateDecision(text) {
    if (!text) return "本轮没有额外决策说明。";
    if (text.includes("Archive improved")) {
      return "Archive 更新为新的最优候选；下一步在利用该候选与继续探索之间切换。";
    }
    if (text.includes("below the current target")) {
      return "候选合法但得分偏低；下一步优先尝试 multi-start 或围绕当前最优进行扰动。";
    }
    if (text.includes("plateau")) {
      return "检测到平台期；围绕 Archive 最优候选扰动，并通过修复保持几何可行。";
    }
    if (text.includes("no improvement")) {
      return "候选合法但没有提升；切换初始化方式或继续扰动当前最优。";
    }
    if (text.includes("Increase safety margin")) {
      return "根据 Evaluator 反馈增加安全边距，并优先使用修复策略。";
    }
    return text;
  }

  function recordsForTask(data, task) {
    const byTask = data.trajectory_by_task && data.trajectory_by_task[task];
    if (Array.isArray(byTask) && byTask.length) return byTask.slice();
    return (data.trajectory || []).filter((record) => String(record.task || "").toUpperCase() === task);
  }

  function lineageForTask(data, task) {
    const lineage = data.lineage && data.lineage[task];
    return Array.isArray(lineage) ? lineage.slice() : [];
  }

  function candidateKey(record) {
    return record && record.candidate_id ? String(record.candidate_id) : "";
  }

  function parentKey(record) {
    const fields = [
      "parent_candidate_id",
      "parent_id",
      "parent",
      "source",
      "previous_candidate",
      "previous_candidate_id",
    ];
    for (const field of fields) {
      const value = record && record[field];
      if (value !== undefined && value !== null && String(value).trim() !== "") return String(value);
    }
    return "";
  }

  function scoreValue(record) {
    const score = Number(record && record.score);
    return Number.isFinite(score) ? score : -Infinity;
  }

  function sumValue(record) {
    const sum = Number(record && record.sum_radii);
    return Number.isFinite(sum) ? sum : -Infinity;
  }

  function iterationValue(record) {
    const iteration = Number(record && record.iteration);
    return Number.isFinite(iteration) ? iteration : 0;
  }

  function compareCandidates(a, b) {
    const validDelta = Number(Boolean(a && a.valid)) - Number(Boolean(b && b.valid));
    if (validDelta !== 0) return validDelta;
    const scoreDelta = scoreValue(a) - scoreValue(b);
    if (Math.abs(scoreDelta) > 1e-12) return scoreDelta;
    const sumDelta = sumValue(a) - sumValue(b);
    if (Math.abs(sumDelta) > 1e-12) return sumDelta;
    return -Math.abs(iterationValue(a)) + Math.abs(iterationValue(b));
  }

  function bestRecord(records) {
    if (!records.length) return null;
    return records.reduce((best, record) => (compareCandidates(record, best) > 0 ? record : best), records[0]);
  }

  function hasParentLinks(records) {
    const ids = new Set(records.map(candidateKey).filter(Boolean));
    return records.some((record) => {
      const parent = parentKey(record);
      return parent && ids.has(parent);
    });
  }

  function buildParentPath(records, best) {
    const byId = new Map(records.map((record) => [candidateKey(record), record]).filter(([id]) => id));
    const path = [];
    const seen = new Set();
    let current = best;
    while (current && candidateKey(current) && !seen.has(candidateKey(current))) {
      path.push(current);
      seen.add(candidateKey(current));
      current = byId.get(parentKey(current));
    }
    return path.reverse();
  }

  function buildIterationPath(records) {
    const grouped = new Map();
    records.forEach((record) => {
      const iteration = iterationValue(record);
      const current = grouped.get(iteration);
      if (!current || compareCandidates(record, current) > 0) grouped.set(iteration, record);
    });
    return Array.from(grouped.entries())
      .sort((a, b) => a[0] - b[0])
      .map((entry) => entry[1]);
  }

  function buildBestPath(data, task) {
    const trajectory = recordsForTask(data, task);
    const lineage = lineageForTask(data, task);
    const allRecords = trajectory.length ? trajectory : lineage;
    const best = bestRecord(allRecords.filter((record) => record.valid)) || bestRecord(allRecords);
    if (!best) {
      return { path: [], best: null, mode: "empty", note: "没有找到可回放的候选记录。" };
    }

    const parentSource = lineage.length > 1 ? lineage : allRecords;
    if (hasParentLinks(parentSource)) {
      const parentBest = bestRecord(parentSource.filter((record) => record.valid)) || bestRecord(parentSource);
      const path = buildParentPath(parentSource, parentBest || best);
      return { path, best: parentBest || best, mode: "parent", note: "已根据父子关系回溯当前最优候选路径。" };
    }

    if (trajectory.length > 1) {
      const path = buildIterationPath(trajectory);
      const pathBest = bestRecord(path.filter((record) => record.valid)) || bestRecord(path);
      return { path, best: pathBest || best, mode: "trajectory", note: "当前数据没有明确父子关系，已按 iteration 从 trajectory 重建回放路径。" };
    }

    const single = lineage.length ? lineage : allRecords;
    const path = single.slice(0, 1);
    return {
      path,
      best: bestRecord(path.filter((record) => record.valid)) || bestRecord(path) || best,
      mode: "single",
      note: `当前 demo 数据中 Task ${task} 仅包含一个可回放节点。`,
    };
  }

  function detailField(label, value, extraClass) {
    return `
      <div class="detail-field ${extraClass || ""}">
        <dt>${escapeHtml(label)}</dt>
        <dd>${value}</dd>
      </div>
    `;
  }

  function renderKpis(data) {
    const root = byId("kpi-grid");
    const taskA = data.tasks && data.tasks.A ? data.tasks.A : {};
    const taskB = data.tasks && data.tasks.B ? data.tasks.B : {};
    const items = [
      ["Task A score", fmt(taskA.score), "矩形圆填充"],
      ["Task B score", fmt(taskB.score), "单位正方形圆填充"],
      ["Combined score", fmt(data.combined_score), "两项平均"],
      ["Archive 记录", String((data.trajectory || []).length), "候选轨迹"],
    ];
    root.innerHTML = items
      .map(
        ([label, value, desc]) => `
          <div class="kpi">
            <div class="kpi-label">${label}</div>
            <div class="kpi-value">${value}</div>
            <div class="kpi-desc">${desc}</div>
          </div>
        `
      )
      .join("");
    const generatedAt = byId("generated-at");
    if (generatedAt) generatedAt.textContent = data.generated_at || "N/A";
  }

  function renderTimeline(data, task) {
    const timeline = byId("timeline");
    const detail = byId("timeline-detail");
    const meta = byId("timeline-meta");
    const panel = byId("trajectory-panel");
    const pathState = buildBestPath(data, task);
    const source = pathState.path;
    const bestId = candidateKey(pathState.best);

    if (meta) {
      meta.innerHTML = `
        <span class="path-note">${escapeHtml(pathState.note)}</span>
        <span class="path-count">${source.length} 个节点</span>
      `;
    }

    if (!source.length) {
      timeline.innerHTML = '<div class="empty-state">没有找到可回放的候选记录。</div>';
      detail.innerHTML = "";
      return;
    }

    timeline.innerHTML = source
      .map((record, index) => {
        const id = candidateKey(record);
        const isBest = id && id === bestId;
        const active = isBest ? " active" : "";
        const currentBest = isBest ? " current-best" : "";
        const stateClass = statusClass(record);
        return `
          <button class="timeline-node${active}${currentBest} ${stateClass}" type="button" data-index="${index}" title="${escapeHtml(id)}">
            <span class="timeline-rail" aria-hidden="true">
              <span class="timeline-dot"></span>
            </span>
            <span class="node-body">
              <span class="node-topline">
                <span class="node-step">iteration ${escapeHtml(record.iteration)}</span>
                ${isBest ? '<span class="best-badge">Best</span>' : ""}
              </span>
              <span class="node-title">${escapeHtml(id)}</span>
              <span class="node-badges">
                <span class="status-badge ${stateClass}">${escapeHtml(statusText(record))}</span>
                <span class="score-badge">score ${fmt(record.score, 6)}</span>
                <span class="strategy-tag">${escapeHtml(translateStrategy(record.strategy))}</span>
              </span>
              <span class="node-decision">${escapeHtml(translateDecision(record.decision))}</span>
            </span>
          </button>
        `;
      })
      .join("");

    function show(index) {
      const record = source[index] || source[source.length - 1];
      const id = candidateKey(record);
      const isBest = id && id === bestId;
      const stateClass = statusClass(record);
      document.querySelectorAll(".timeline-node").forEach((node) => node.classList.remove("active"));
      const activeNode = timeline.querySelector(`[data-index="${index}"]`);
      if (activeNode) activeNode.classList.add("active");
      detail.innerHTML = `
        <div class="detail-heading">
          <div>
            <span>${task === "A" ? "Task A" : "Task B"} Candidate Detail</span>
            <strong title="${escapeHtml(id)}">${escapeHtml(id)}</strong>
          </div>
          <div class="detail-badges">
            ${isBest ? '<span class="best-badge">Current Best</span>' : ""}
            <span class="status-badge ${stateClass}">${escapeHtml(statusText(record))}</span>
          </div>
        </div>
        <dl class="detail-grid">
          ${detailField("iteration", escapeHtml(record.iteration))}
          ${detailField("valid", `<span class="status-badge ${stateClass}">${escapeHtml(boolText(record.valid))}</span>`)}
          ${detailField("score", `<span class="detail-score">${fmt(record.score)}</span>`, "score-field")}
          ${detailField("sum_radii", escapeHtml(fmt(record.sum_radii)))}
          ${detailField("strategy", `<span class="strategy-tag">${escapeHtml(translateStrategy(record.strategy))}</span><span>${escapeHtml(strategyCode(record.strategy))}</span>`)}
          ${detailField("failure_type", `<span class="failure-pill ${stateClass}">${escapeHtml(translateFailure(record.failure_type))}</span><span>${escapeHtml(record.failure_type || "none")}</span>`)}
          ${detailField("candidate_id", `<span class="detail-code">${escapeHtml(id)}</span>`, "wide")}
          ${detailField("source file", `<span class="detail-code">${escapeHtml(record.code_snapshot || "N/A")}</span>`, "wide")}
        </dl>
        <div class="decision-block">
          <span>decision</span>
          <p>${escapeHtml(translateDecision(record.decision))}</p>
        </div>
      `;
    }

    timeline.querySelectorAll(".timeline-node").forEach((node) => {
      const index = Number(node.dataset.index);
      node.addEventListener("click", () => show(index));
      node.addEventListener("mouseenter", () => show(index));
    });
    const defaultIndex = Math.max(0, source.findIndex((record) => candidateKey(record) === bestId));
    if (panel) panel.dataset.pathMode = pathState.mode;
    show(defaultIndex);
  }

  function renderCircleSvg(task) {
    const width = Number(task.width || 1);
    const height = Number(task.height || 1);
    const pad = Math.max(width, height) * 0.05;
    const viewBox = `${-pad} ${-pad} ${width + pad * 2} ${height + pad * 2}`;
    const circles = (task.centers || [])
      .map((center, index) => {
        const r = Number((task.radii || [])[index] || 0);
        const color = colors[index % colors.length];
        const opacity = 0.20 + Math.min(0.28, r * 1.5);
        return `<circle cx="${center[0]}" cy="${center[1]}" r="${r}" fill="${color}" fill-opacity="${opacity}" stroke="${color}" stroke-width="${Math.max(width, height) * 0.003}" />`;
      })
      .join("");
    return `
      <svg class="packing-svg" viewBox="${viewBox}" role="img" aria-label="${task.label} 圆填充图" preserveAspectRatio="xMidYMid meet">
        <defs>
          <pattern id="grid-${task.label.replace(/\s+/g, "-")}" width="${width / 8}" height="${height / 8}" patternUnits="userSpaceOnUse">
            <path d="M ${width / 8} 0 L 0 0 0 ${height / 8}" fill="none" stroke="#dbeafe" stroke-width="${Math.max(width, height) * 0.0015}" />
          </pattern>
        </defs>
        <rect x="0" y="0" width="${width}" height="${height}" rx="${Math.max(width, height) * 0.012}" fill="#f8fbff" stroke="#93c5fd" stroke-width="${Math.max(width, height) * 0.004}" />
        <rect x="0" y="0" width="${width}" height="${height}" fill="url(#grid-${task.label.replace(/\s+/g, "-")})" opacity="0.65" />
        ${circles}
      </svg>
    `;
  }

  function renderTaskResult(id, task) {
    const root = byId(id);
    root.innerHTML = `
      <div class="result-visual">${renderCircleSvg(task)}</div>
      <div class="metric-grid">
        <div><span>半径和</span><strong>${fmt(task.sum_radii)}</strong></div>
        <div><span>官方得分</span><strong>${fmt(task.score)}</strong></div>
        <div><span>width / height</span><strong>${fmt(task.width, 6)} / ${fmt(task.height, 6)}</strong></div>
        <div><span>最小圆间安全边距</span><strong>${sci(task.min_pairwise_margin)}</strong></div>
        <div><span>最小边界安全边距</span><strong>${sci(task.min_boundary_margin)}</strong></div>
        <div><span>导出文件路径</span><strong>${task.export_path}</strong></div>
      </div>
    `;
  }

  function renderResults(data) {
    const tasks = data.tasks || {};
    renderTaskResult("result-a", tasks.A || {});
    renderTaskResult("result-b", tasks.B || {});
  }

  function renderScoreChart(data) {
    const root = byId("score-chart");
    const byTask = data.trajectory_by_task || {};
    const series = ["A", "B"].map((task, idx) => ({
      task,
      color: colors[idx],
      points: (byTask[task] || []).map((record, index) => ({
        x: index,
        y: Number(record.score || 0),
        label: record.candidate_id,
      })),
    }));
    const all = series.flatMap((item) => item.points);
    if (!all.length) {
      root.innerHTML = '<div class="empty-state">没有可绘制的 score trajectory。</div>';
      return;
    }
    const maxLen = Math.max(...series.map((item) => Math.max(1, item.points.length - 1)));
    const minScore = Math.min(...all.map((item) => item.y), 0.72);
    const maxScore = Math.max(...all.map((item) => item.y), 1.0);
    const w = 720;
    const h = 260;
    const pad = 34;
    const yScale = (score) => h - pad - ((score - minScore) / Math.max(0.001, maxScore - minScore)) * (h - pad * 2);
    const xScale = (x) => pad + (x / Math.max(1, maxLen)) * (w - pad * 2);
    const paths = series
      .map((item) => {
        if (!item.points.length) return "";
        const d = item.points
          .map((point, index) => `${index === 0 ? "M" : "L"} ${xScale(point.x).toFixed(2)} ${yScale(point.y).toFixed(2)}`)
          .join(" ");
        const dots = item.points
          .map((point) => `<circle cx="${xScale(point.x)}" cy="${yScale(point.y)}" r="4.5" fill="${item.color}"><title>${item.task}: ${point.label} ${fmt(point.y)}</title></circle>`)
          .join("");
        return `<path d="${d}" fill="none" stroke="${item.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />${dots}`;
      })
      .join("");
    root.innerHTML = `
      <svg class="chart-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="score trajectory">
        <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#bfdbfe" />
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="#bfdbfe" />
        <text x="${pad}" y="${pad - 10}" fill="#64748b" font-size="13">${fmt(maxScore, 3)}</text>
        <text x="${pad}" y="${h - 10}" fill="#64748b" font-size="13">${fmt(minScore, 3)}</text>
        ${paths}
      </svg>
      <div class="chart-legend">
        <span><i style="background:${colors[0]}"></i>Task A</span>
        <span><i style="background:${colors[1]}"></i>Task B</span>
      </div>
    `;
  }

  function renderBars(rootId, rows, valueSelector, labelSelector, metaSelector) {
    const root = byId(rootId);
    if (!rows.length) {
      root.innerHTML = '<div class="empty-state">暂无统计数据。</div>';
      return;
    }
    const maxValue = Math.max(...rows.map(valueSelector), 1);
    root.innerHTML = rows
      .map((row, index) => {
        const value = valueSelector(row);
        const width = Math.max(4, (value / maxValue) * 100);
        return `
          <div class="bar-row">
            <div class="bar-label">${labelSelector(row)}</div>
            <div class="bar-track"><span style="width:${width}%; background:${colors[index % colors.length]}"></span></div>
            <div class="bar-value">${metaSelector(row)}</div>
          </div>
        `;
      })
      .join("");
  }

  function renderStats(data) {
    const strategyRows = Object.entries(data.strategy_stats || {}).map(([strategy, stat]) => ({
      strategy,
      stat,
    }));
    renderBars(
      "strategy-bars",
      strategyRows,
      (row) => Number(row.stat.best_score || 0),
      (row) => `<span class="bar-title">${translateStrategy(row.strategy)}</span><span class="bar-sub">${row.strategy}</span>`,
      (row) => `${row.stat.attempts} 次 / best ${fmt(row.stat.best_score)}`
    );

    const failureRows = Object.entries(data.failure_stats || {}).map(([failure, count]) => ({
      failure,
      count,
    }));
    renderBars(
      "failure-bars",
      failureRows,
      (row) => Number(row.count || 0),
      (row) => `<span class="bar-title">${translateFailure(row.failure)}</span><span class="bar-sub">${row.failure}</span>`,
      (row) => `${row.count} 次`
    );
    renderScoreChart(data);
  }

  function renderSubmissionTree(data) {
    const tree = byId("submission-tree");
    tree.textContent = (data.submission_tree || []).join("\n");
  }

  function setupControls(data) {
    const tabs = document.querySelectorAll("[data-timeline-task]");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        tabs.forEach((item) => item.classList.remove("active"));
        tab.classList.add("active");
        renderTimeline(data, tab.dataset.timelineTask);
      });
    });

    const start = byId("start-replay");
    if (start) {
      start.addEventListener("click", () => {
        document.querySelector("#replay").scrollIntoView({ behavior: "smooth", block: "start" });
        const active = document.querySelector("[data-timeline-task].active");
        renderTimeline(data, active ? active.dataset.timelineTask : "A");
      });
    }
  }

  function init() {
    const data = demoData();
    renderKpis(data);
    renderTimeline(data, "A");
    renderResults(data);
    renderStats(data);
    renderSubmissionTree(data);
    setupControls(data);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
