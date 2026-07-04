# Optional FICO Benchmark Seed

Source:

FICO public `newsolutions.txt`, Problem 13.

This benchmark source is optional in the Agent. If a local public copy is placed at either of these paths, `optional_fico_task_a_seed` will parse it as a Task A warm-start candidate:

- `benchmarks/fico/problem13_task_a.txt`
- `benchmarks/fico/newsolutions.txt`

The seed is never accepted directly. The Agent converts it into a static candidate and the official `task_A/evaluate.py` must pass before it can replace the best valid archive entry.

No FICO coordinate seed is bundled here because the public `newsolutions.txt` URL was not available in the local repository at implementation time.
