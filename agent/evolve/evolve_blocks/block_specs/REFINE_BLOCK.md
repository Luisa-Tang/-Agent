# REFINE Block

Allowed edits:
- small coordinate descent;
- repair/polish through context helper functions;
- contact graph preserving or breaking refinement with bounded steps.

Forbidden edits:
- long unbounded optimization;
- subprocesses or network calls;
- official evaluator calls.

Inputs:
- centers, radii, task/container, context budgets.

Outputs:
- refined centers/radii and, for Task A, width/height.

Verification:
- internal geometry, novelty, official evaluator.

Red flags:
- unbounded loops;
- large moves that destroy all active constraints.
