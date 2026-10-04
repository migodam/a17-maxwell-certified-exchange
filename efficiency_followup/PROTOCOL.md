# A17 efficient verification — execution contract

Created 2026-10-04 (Asia/Singapore). This is a separate efficiency follow-up, authorized by the user's A17-Efficient Verification request. It does not reopen or overwrite EXCHANGE_R1, CLOSURE_R1, their rules/results, or public commit f5bd7d0aa0e229110c425f1becb6384e09663196.

## Fixed scientific target

Receiver backbone -> conditional exchange -> actual finite material displacement -> full directional verification -> frozen full-GN material-step fidelity. The fixed metric is R_GN = Phi_full(s_U)-Phi_full(s_*) = 0.5 ||s_U-s_*||_H^2. New policies are separately named deployment experiments, never substitutes for the original results. No new ranking theory, selector, PCG/preconditioner, NN/GNN/Flow or A18 model training.

## Required order

1. Audit saved runtime evidence and actual imported backend before optimization. Attribute exclusive times, counts and missing measurements explicitly; never sum nested timers.
2. Saved-array pilot: g_F availability at the actual verification base point; exact upper-bound screening; physical displacement ranks (same round/state, with cross-state coordinates mapped explicitly); original anchor defects and potential policy paths.
3. Only justified implementation changes in this directory: exact screening, causal within-state tangent reuse, genuine block multi-RHS, bounded verification-frequency policies and previous-current-space reuse. Preserve FP64, physical material gauge, polarizability derivative, real material pullback, regularization, constraints and line search.
4. Pilot objects fixed before results: smooth voxel 2002 and shell Gaussian 2009, states early/middle/late, shared complex current ranks 4/8/16. Near-contact Gaussian 2005 is optional and never replaces a failed pilot. First measured profiling can use their middle states.
5. Gate A: <=2 full Jd directions per outer state and >=90% original frozen-GN gap reduction on each pilot object -> GO; 2–3 directions and 80–90% -> CONDITIONAL; otherwise NO-GO for the tested branch. Savings must include any new adjoint, Jx, candidate work, screening and fallbacks.
6. Only after Gate A, bounded matched nonlinear comparisons. Gate B requires preserved accepted-update stability and no material/held-out degradation beyond a predeclared quality tolerance; lock that tolerance before new outcomes. Gate C target exchange/receiver <=1.3, stronger <=1.15; >1.5 needs remaining-bottleneck explanation. No end-to-end speed claim from a kernel benchmark.
7. Empirical anchor-radius triage is distinct from analytic pruning and is not certified. Cheap-reject first; no approximate accept in the recommended policy. Verification ROM only if earlier routes do not substantially reduce Jd, with charged full fallback and no enlarged research scope.

## Mathematical and causal boundaries

Q(d;x) = -g_F(x)^T d -0.5||J_F d||^2 -0.5 d^T Lambda d. Q_upper0 = -g_F(x)^T d -0.5 d^T Lambda d is an exact algebraic upper bound only when the gradient is at x, not merely at outer-state step zero. An extra full adjoint to obtain it is not free. A numerical tolerance is not a deterministic output-error certificate; absent Jd, a nonnegative predeclared tolerance lower bound may be used for conservative rejection.

J_F is fixed only inside one frozen linearization. QR reuse is causal: future displacements depend on accepted exchanges and cannot be acquired free in advance. Cross-state material-vector alignment is a rank diagnostic, not permission to reuse stale Jd. Every accepted change recomputes conditional proposals. All finalist ranks and full physical RHS are separately counted.

## Resources and isolation

Do not reset the earlier authorized cumulative 12-GPU-hour ceiling. Closed charged occupation is 19,962.060 s; at most 23,237.940 s (6.455 h) remain unless the user authorizes more. New attempts, failed jobs, profiling and retries are charged in a separate ledger linked to that carry. A pilot failing its gate stops its dependent branch; unused budget is not a requirement.

One numerical GPU process, no intervention in other research jobs, no old queue restart. Before each launch inspect actual host/GPU/process/lock availability and source hashes. Per job <=2 h; 80% VRAM warning, 90% stop. CPU saved-array work is separately timed. No framework/install/security changes, hidden precision or grid changes, or automatic retries.

Root Codex owns algebra, science, gates and interpretation. Bounded authorized GPT-6.1 Sol workers may inspect source, analyze saved arrays, implement known batching/policies and review; their outputs live in research/delegated/a17_eff_* and they cannot launch remote jobs or tune on held-out outcomes.

## Deliverables

A17_EFFICIENCY_PROFILE.md; A17_ANALYTIC_SCREENING.md; A17_DISPLACEMENT_RANK_AUDIT.md; A17_BLOCK_SOLVE_AUDIT.md; A17_VERIFICATION_FREQUENCY_ABLATION.md; A17_STATE_REUSE.md; A17_MULTIFIDELITY_VERIFIER.md; A17_VERIFICATION_ROM.md; A17_EFFICIENCY_PARETO.md; A17_TO_A18_EFFICIENCY_INTERFACE.md; A17_EFFICIENT_VERIFICATION_FINAL.md. Every branch has GO/CONDITIONAL/NO-GO or a explicit NOT_RUN reason. Final report recommends exactly one policy, retaining the original if no improvement passes.

Also retain code/config/input hashes, cost and failure receipts, source identities, per-candidate data and plotting/reproduction commands. A18 gets only interfaces/features/true-vs-anchor pairs and timing/fallback labels; no network is trained.

After each compaction reread this file and RESUME_STATUS.md before proceeding. Initial status: SAVED_EVIDENCE_AND_RUNTIME_AUDIT; no new numerical job has started.
