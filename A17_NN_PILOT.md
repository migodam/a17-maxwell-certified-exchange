# Bounded learned priority

**All six models retain much of the held-out opportunity; none exceeds the analytical anchor. No learned contribution or net speedup is retained.**

Frozen MLP/group mean, 3 seeds each, 100 epochs in FP64 on CPU, run once. Training uses near-contact + layered objects; validation uses smooth; testing uses asymmetric. Objects are never split. The 47712 records/43108 feasible labels are indexed and source-hashed.

|Model|seed 1 held-out top 2|seed 2|seed 3|
|---|---:|---:|---:|
|MLP|.988459|.988461|.991617|
|Group mean|.765571|.759893|.980288|
|Analytical anchor|.998850|.998850|.998850|

This evaluates already-paid top 2 labels/no-op, not fresh physical acceptance or nonlinear reconstruction. All six runs are preserved. The 70% capture gate is met; the required ≥2× all-in cost gain is not measured.

Features come from reduced endpoints and anchors, not full Evaluator information. All 36 forbidden-field canaries PASS; the ownership interface is not a security sandbox.

CPU training takes 9.413 s; inference at all 36 checkpoints with six models takes .02453 s. Physical feature acquisition and new validation are excluded from those numbers, and speedup stays null.

