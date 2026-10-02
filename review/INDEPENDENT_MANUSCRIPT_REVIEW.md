# A17 final manuscript independent review

Read-only review of the current English paper, professor report, Chinese reader guide and exchange supplement, from TAP, electromagnetic-subspace and numerical-linear-algebra perspectives. No physics, source, data or manuscript was changed. Evidence was limited to the owner metrics, final pilot tables, information-ranking outputs, matched timing report, tiny pilot summary and final figures. Earlier historical numbers were checked for scope and consistency of presentation, not independently re-audited against their original archives.

**Overall:** the central positive result is supported and should remain prominent: at fixed actual current dimension, directed-verified exchange improves 35/36 correlated state–budget cells, with no resolved deterioration and a 65.2152% median reduction of object-summed full-GN gap. The feedback → material curvature plus normal offset → actual signed utility chain is coherent. No fatal contradiction was found in the reviewed exchange algebra. Three local clarifications are recommended; a fourth causal interpretation remains OPEN rather than a demonstrated numerical error.

## Findings requiring a small edit

### F1 — Object-summed capture is presented as a checkpoint-level rate in the reader guide (P2, high confidence)

`EXCHANGE_READER_EVIDENCE.tex:42` defines a single checkpoint's best gain `Q_max` and then says the anchor-selected action obtains 98.1–99.6% of it. That range is instead **four object-level ratios of sums over nine receiver checkpoints**, not a per-checkpoint lower bound. `CONDITIONAL_EXCHANGE_RESULTS.tex:21` correctly says “object-summed best one-decision gain.” The same denominator should be explicit in `DISCUSSION.tex:8`, `EXCHANGE_CHEN_RESULTS.tex:12`, `ABSTRACT.tex:2`, and `READER_GUIDE.tex:10`; these shorter statements are ambiguous rather than separately false.

Concrete counterexample from `analysis_output_final/same_checkpoint_score_regret.csv`: the smooth object, middle state, rank 16 has teacher best gain `2.443137278800288e-7`, full-dictionary anchor top-one actual gain `-4.774163444385188e-7`, and numerical tolerance `2.1871235165076276e-18`. Its mandatory top-one capture is `-1.9541118241`, not 98.1–99.6%. This does not contradict the strong object-summed result or deployed acceptance: that aggregate is gain-weighted, and verification includes rejection/no-op.

Suggested concise replacement: “在每个对象的九个共同 receiver 起点上，先把单次收益求和再相除，完整字典 anchor top-one 的对象级捕获率为 98.1–99.6%；它不是每个 checkpoint 的保证。” Define the ratio as `sum_c Q_anchor,c / sum_c max(0,Q_best,c)` and retain the distinction from multi-action path capture.

### F2 — Numerical acceptance sentence has an overbroad policy scope (P2, high confidence)

`DISCUSSION.tex:20`: “Every reported exchange passes the frozen numerical acceptance rule.” Multiple reported policies use different permissions and acceptance semantics. In `analysis_output_final/per_policy_main.csv`, the unchecked `surrogate_exchange` accepts 83 actions, **8 with actual complete gain no greater than their tolerance**; `directed_verified` accepts 75, all above their complete-gain tolerance. The offline teacher accepts 105 positive actions. Thus an unqualified “every reported exchange” can give the unchecked comparator the deployed method's guarantee.

Replace with “Every accepted directed-verified exchange passes the frozen complete-gain numerical rule.” Keep the following sentence about missing certified output-error bounds. The main result, no-op protection and 22 unresolved/14 budget stops remain intact.

### F3 — “Reference step” update is ambiguous against the frozen reference contract (P2 clarification, medium-high confidence)

`DISCUSSION.tex:5` says the “reference step” is updated after accepting a move. The full target and richer anchor are fixed within a frozen state; the **current endpoint step**, conditional factors and cached output change. `EXCHANGE_READER_EVIDENCE.tex:5` explicitly states the complete reference remains unchanged; `EXCHANGE_DERIVATIONS_ZH.tex:63` correctly says to update the space, material step and cached output. Calling the changing endpoint a reference step can suggest the scoring reference or full target is adapted along the path.

Replace “reference step” with “current endpoint step and cached output.” No change in algorithm or result is requested.

## OPEN causal wording for owner review

### F4 — The information witness proves insufficiency, but not an isolated offset cause (medium confidence)

`EXCHANGE_CHEN_RESULTS.tex:21` says the missing element is the offset's signed residual alignment; `EXCHANGE_READER_EVIDENCE.tex:72` says the formula “points to the cause.” The witness establishes that information magnitude and the selected curvature-fidelity scalar can improve while utility worsens. The exact formula contains **both** `Delta h` and `Delta I s_o`, filtered by `H_n^{-1}`; the permitted final evidence does not report a controlled decomposition identifying which term dominates this particular witness. The English `CONDITIONAL_EXCHANGE_RESULTS.tex:28` is appropriately framed as a missing residual-dependent term, rather than a sole-cause attribution.

Suggested strong formulation: “这个见证说明信息幅度和曲率保真度仍不足以决定收益；实际步还由残差相关的 offset 与曲率改变共同决定。” Preserve the concrete positive-information/negative-utility witness. This is a causal-scope clarification, not a reason to remove it or a verdict that its physical explanation is wrong.

## Checks that support the present narrative

| Question | Finding and evidence |
|---|---|
| 35/36 denominator | Owner metrics: 36 cells = four objects × three saved states × three current budgets. 35 positive, 36 nonworse; `CONDITIONAL_EXCHANGE_RESULTS.tex:4–6`, professor results:2, reader evidence:7/35 correctly identify correlation and aggregation. |
| 65.2% denominator | Ratios independently recomputed from `object_summed_ratios.csv`: 0.1904004867, 0.6537637412, 0.2776323739, 0.4180635909. Median 0.3478479824, reduction 0.6521520176. This is not pooled-cell improvement, a median cell ratio, or reconstruction accuracy. |
| Single action versus path | English results:14/21 distinguish object-summed same-checkpoint capture from verified/teacher three-action path gain capture. The 89.8/43.4/85.2/69.5% path figures are not score-only regret. Shortpool opportunity is explicitly conditional on the common receiver checkpoint. |
| Direction and curvature | Core theory:124–162 has the exact two-channel step-change identity and the indefinite curvature-difference example. Exchange theory:20/35 evaluates actual displacement and the complete quadratic. Supplement:34–40 records mandatory top-one negatives 29/36 for actual-direction linear scoring versus 1/36 for quadratic scoring; no-op is absent only in that diagnostic. Earlier 0.1171 entire-path ratio is correctly labelled confounded. |
| Physical chain and literature boundary | The paper distinguishes direct radiation from retained-space feedback, includes the material injection derivative and shared real material coordinates, and acknowledges SOM/TSOM internal propagation. Paired support is not presented as an optimal error-correction method; its standard material-Krylov control remains stronger. No novelty verdict is supplied by this review. |
| Information and fidelity | Fixed P=1e-5 is separated from LM. The witness has Delta volume 6.3127257611, Delta effective dimension 1.8613056862, Q=-5.5753875177e-5; its gap increases from 1.4968486970e-4 to 2.0543874488e-4. The sidecar links the physical action and spectrum. Complete Gaussian versus 16-direction voxel PROJECTED_ONLY scope is explicit. Information rankings cover 3665 persisted spectrum records, with utility-biased teacher augmentation separated, not the entire teacher neighborhood. |
| Weak directions | Supplement:54 correctly reports zero full-reference weak directions for all six Gaussian states under nu<=1 and P=1e-5; early states have 2 intermediate/52 strong, later states 0/54. This says nothing about complete voxel weakness. Risk decomposition retains LM and the reference-gradient cross term. |
| Certification | Exchange theory:39–43 and supplement:17–25 require legitimate output bounds, keep deterministic radius null, and distinguish candidate acceptance from all-neighborhood stopping. No shortlist is certified complete. |
| Candidate permissions | Protocol separates local exhaustive teacher/evaluator data from reduced-anchor ranking and top-two physical actions. The tiny pilot has 36 round-0 dictionaries, 47712 records/43108 feasible, six retained runs and legal reduced-model trace features. All learned heldout captures trail the analytical anchor top-two 99.885%; no learned speedup or new physical validation is claimed. |
| Timing | Warm physical comparisons cover two representative early states and five rotations, not all objects or independent scientific replicates. Conditional 0.594–3.400 s is slower than cached one-pass 0.113–0.219 s; cold/all-in acceleration remains unmeasured. Historical setup, failed attempts and offline evaluations are separately charged. The statement in discussion:18 about avoiding a complete Jacobian is an action-count/scaling explanation; it should not be read as a measured full-J construction speedup. |
| Professor report and undefined terms | The report is first-person and addresses the professor directly. It defines Gaussian/N-Port, SOM/TSOM, material injection, retained feedback, offset and actual gain before using them. English technical terms are retained as requested. The repeated motivation paragraphs are an editorial choice, not a scientific bug. |

## Boundaries

No request to weaken the demonstrated fixed-budget improvement, remove the information reversal/pair witnesses, or replace direct positive statements with generic caution. No new certificate, acceleration, nonlinear reconstruction or novelty judgment follows from this review. Exact historical A16 physical numbers and bibliography were not independently revalidated under the allowed evidence scope. Only the three small scope/denominator clarifications above are recommended before final owner review; F4 remains an interpretation decision.
