# Small Maxwell implementation gate

CPU checkpoint v2: PASS. Tiny 3^3 voxel geometry, six illuminations, 54 shared real material coordinates. No holdout scientific result or imaging/certification claim. CUDA NOT_RUN; implementation supports explicit --device cuda only, comparing identical inputs with CPU on the tiny physical model and direct/Woodbury EndpointBatch and CoreEndpointBatch branches.

Smoke SHA256: `8b8716163b9ba63cb17cbfe0c45388f92dd318bf61370c91adda7631504304ae`
Online tolerance SHA256: `0cb18fc4b91f377c8a1cc96c426bbcff7e000a846b515aa53e59a384466cb8bf`
Verified exchange SHA256: `b237b985377831eb9c93b36449ce84493494ca373769fe74acc0551c1ec421de`
Baseline manifest SHA256: `b47422c1ec7c945a6cea536ec479c3515ab322a631d088d62eed25c2655e34db`

|Check|Relative error|
|---|---:|
|real_adjoint|1.5544023e-15|
|full_material_jvp|6.97488429e-16|
|polarizability_derivative_finite_difference|7.17968337e-10|
|Schur_swap_vs_direct_endpoints|1.02154788e-15|
|existing_vs_independent_base|7.52028403e-16|
|existing_vs_independent_child|7.27434769e-16|
|actual_step_change|1.86714572e-15|
|phase_permutation_step|1.36754617e-16|
|directed_gain_vs_objective_difference|2.78064235e-14|
|online_gain_vs_objective_difference|2.78064235e-14|
|online_gain_vs_legacy_directional|0|
|direct_core_vs_direct_steps|8.80917903e-17|
|direct_core_vs_direct_j_steps|3.34597597e-16|
|direct_core_vs_direct_info_trace|1.73478965e-16|
|woodbury_core_vs_direct_steps|6.27855215e-17|
|woodbury_core_vs_direct_j_steps|3.11948475e-16|
|woodbury_core_vs_direct_info_trace|1.73478965e-16|
|CPU_direct_vs_woodbury_steps|3.61629026e-16|
|CPU_direct_vs_woodbury_j_steps|4.90452752e-16|
|CPU_direct_vs_woodbury_info_trace|0|

All errors below the unchanged 1e-9 implementation threshold. Finite-difference derivative uses centered epsilon=1e-5, and is the largest error. Numerically exact identities are near floating-point precision.

Online Q absolute discrepancy: 3.17129135452e-18; tau: 9.7332553499e-13. Positive no-op tolerance and zero no-op gain checked. This is a rounding-significance check, not a certified Maxwell output radius.

Input-array SHA256, full pinned runtime receipt, physical counters and both endpoint branch statistics are in small_maxwell_smoke.json; smoke_inputs.npz preserves this tiny fixture. v1 was retained; v2 adds explicit error-within-tau admission and disables bytecode writes.

Invocation from project root:

```text
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 Gaussian/.venv_nn/bin/python Gaussian/A17/CLOSURE_R1/tests/small_maxwell_smoke.py --baseline-root Gaussian/A17/EXCHANGE_R1 --out NEW_OUTPUT_DIRECTORY --device cpu
```

Output must be fresh and outside the immutable baseline. No remote/GPU operation was performed.
