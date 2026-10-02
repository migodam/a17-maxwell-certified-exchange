# Paired information rankings from persisted diagnostics

CPU read-only postprocessor. Input: analyzer `move_information.csv`, `diagnostic_coverage.csv`, and each source checkpoint's original teacher JSONL, information JSONL, information coverage, receiver baseline and numerical tau. No fullJ is read; no physics/action, GPU, remote operation, selector/source/data modification or certificate is produced.

Current development run: 18 checkpoints, 1949 linked spectrum moves; resolved18/18, CPU approximately3seconds. Fixed12 incoming per outgoing/removal is analyzed separately from the spectrum subset augmented by an optional teacher-best move. The latter is explicitly utility-selected and biased. Neither subset covers the full teacher domain; correlations and regret here cannot represent its full information ranking.

Metrics: receiver_score, delta_logdet_volume, delta_effective_dim, Qhat; information-fidelity improvement only when original baseline and child spectral fidelity are both available in matching full/projected scope. Its score is baseline fidelity minus child fidelity (smaller raw discrepancy is better). No missing fidelity is reconstructed, no absolute information magnitude is mistaken for fidelity, no full material spectrum inferred from public probes.

For each metric, only rows with finite metric and Q are paired; null exclusions and constants remain explicit. Kendall tau-b and Spearman use average ranks after numerical tie grouping. Q/Qhat tolerance is original checkpoint tau. Other metric tie tolerance is128×float64epsilon×max(1,max absolute score), recorded per checkpoint; it is floating diagnostic resolution, not a physical/statistical bound. Tie clusters compare against the first value, avoiding chained ties over a wider gap. Actual scores/Q values are never rounded or overwritten.

Top2 overlap counts candidate-only IDs after original move_index tie-break, with expanded cutoff tie sets also saved. No-op-aware regret compares max(0,max paired Q) with the Q of the score-selected candidate if its marginal score exceeds its recorded tolerance, otherwise no-op0. All marginal scores use no-op zero. A positive proposed score can select a truly negative-Q candidate; its regret is retained. Top-score tied choices expose descriptive minimum/maximum regret, explicitly not uncertainty intervals or certificates. Nearzero positive teacher Q below tau is preserved as a raw tiny value rather than promoted to resolved opportunity.

`CHECKPOINT_SUMMARIES.csv`/`RESULTS.json` report each regime and metric. `OBJECT_DESCRIPTIVE_SUMMARIES.csv` aggregates correlated checkpoints descriptively; summed checkpoint regret is not trajectory regret, and checkpoints/budgets are not independent samples. No inferential p-values, bootstrap/deterministic intervals or scientific mechanism judgment.

`LINKED_MOVES.json` retains state/pool/base-U/logger-U/step/move hashes, original atom IDs, raw teacher/info JSON-record and file hashes, source CSV, receiver/checkpoint JSON and stored delta_b/public_workspace array paths. Teacher raw-array U hash and logger shape/dtype-aware U hash have different definitions and are explicitly identified. The canonical move hash is newly derived from state/pool/base/IDs/drop/add for linkage, not an original physical action digest. Original private execution array paths are retained as provenance, with local array links separately supplied; missing arrays are not rebuilt.

`REVERSAL_WITNESSES.json` saves fixed-subset moves with positive information change and Q<−tau, plus up to3 inverse-information/utility pairs per checkpoint/metric selected by absolute Q separation. Each pair states the total inversion count and illustrative selection bias. These are linked persisted Maxwell labels, not new Maxwell experiments; they do not prove a population frequency or causal explanation. Extra teacher-best records remain available in linked data but do not contaminate fixed-subset witness statistics.

CSV field limits: CSV alone lacks actual atom IDs/drop/add, child Uhash, original numeric tau, persisted full/projected baseline fidelity, fixed incoming IDs and raw delta_b array links. These are read from original checkpoint files; missing/inconsistent sources produce UNRESOLVED. The original CSV cannot support full-neighborhood information ranks or complete material weak-direction claims. Frozen state/pool/base hash, Q/task/receiver scores, exact stored information deltas and spectrum flag are validated against originals before computing summaries.

Run with existing Gaussian Python environment:

```
Gaussian/.venv_nn/bin/python research/delegated/a17_information_rankings/test_ranking_core.py
Gaussian/.venv_nn/bin/python research/delegated/a17_information_rankings/postprocess_rankings.py --input-dir research/delegated/a17_pilot_analysis/analysis_output_development --out research/delegated/a17_information_rankings/development_results
```

The same interface accepts final four-object analyzer outputs after synchronization; no implicit inference of missing states. Seven independent known-order/no-op/constant/null/tie examples PASS. Initial test expected literal Kendall1 where SciPy returned0.9999999999999999; assertion corrected to1e-14 equality tolerance, not score/physical tolerance. An initial NumPy scalar boolean JSON serialization issue was fixed by Python-scalar conversion. Both are mechanical issues, not scientific results; original test failure retained. Old source changes and S8 missing-span records remain untouched. Root owns final interpretation.
