# Human-Agent Division Audit

## Human Provided
- Problem understanding and scoring constraints Evidence: task_A/task_description.md, task_B/task_description.md
- Architecture preferences: no large frameworks, no role-play personas, preserve official evaluators Evidence: submission/report.md, agent/run.py
- Permission to use public benchmark seed data Evidence: benchmarks/dominikkamp/SOURCE.md

## Agent Completed
- Candidate code generation and static export Evidence: task_A/solution.py, task_B/solution.py
- Official evaluator invocation and raw output archival Evidence: submission/agent/run_archive.jsonl, agent/archive/metrics/run_log.jsonl
- Benchmark seed conversion and metadata tracking Evidence: agent/benchmark_seeds.py, benchmarks/dominikkamp/
- Best-valid export, lineage DAG, strategy portfolio, and safety audit Evidence: agent/archive/lineage/task_A_best_lineage.json, agent/archive/lineage/task_B_best_lineage.json, agent/archive/metrics/strategy_portfolio.json, submission/safety_report.json

## Best Candidates
- Task A: `A_000_benchmark_seed_dominikkamp`
- Task B: `B_000_benchmark_seed_dominikkamp`
