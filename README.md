# pilotJSP
Code repository to accompany LION20 conference paper: Learning from Expert Optimization: Expert‑Like Lookahead Policies via Pilot Heuristics

The project implements an active imitation learning pipeline (DAgger) that:

1. **Generates benchmark JSP instances** from the OR-Library style generator (Beasley).
2. **Queries an optimal MIP expert** (Gurobi 13.0) to obtain ground-truth dispatching decisions.
3. **Extracts 13 classical dispatching-rule features** at every decision point (100 steps in a 10×10 JSP).
4. **Builds pairwise preference datasets** using ordinal regression targets.
5. **Learns a dispatching model** that can act as a *pilot heuristic* inside a limited lookahead procedure.
