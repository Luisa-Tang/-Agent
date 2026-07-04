# Static Export Skill

Purpose: export the best valid candidate as a standalone `solution.py` with no network or LLM dependency.

Use when:
- the archive contains at least one official-evaluator-valid candidate;
- the manager loop terminates by iteration budget or time budget;
- submission artifacts are being assembled.

Procedure:
1. Select the best valid candidate by official evaluator score.
2. Copy the archived code snapshot into the task directory's `solution.py`.
3. Ensure arrays are literal NumPy arrays and helper functions are included locally.
4. Re-run the official task evaluator and `evaluate_all.py solution.py`.
5. Copy final solutions, logs, archive summary, and report into `submission/`.

Safety rules:
- Do not require SciPy, LLM APIs, or network access at evaluator import time.
- Keep a fallback safe grid available if no optimized valid candidate exists.
