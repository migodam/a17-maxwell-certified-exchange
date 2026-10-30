# Web-GPT review entry: six 3D reconstruction examples

This is an additive application study, separate from the frozen A17 paper. Read START_HERE.md and RESEARCH_REPORT_ZH.md first, then data/analysis/FINAL_METRICS.csv and the raw real/imaginary slices.

- 6 fixed geometries × 6 policies; 36 primary trajectories, all18 accepted updates/MAX_UPDATES. Additional posthoc full-gradient controls on digit3 and torus, also18 updates.
- True 3D vector-Maxwell DDA, same truth/inverse12³, 12 shared-material illuminations,64 dual-polarized receivers,80 held-out positions,1% fixed complex noise per object. Uniform initial material; no network or truth support.
- Receiver, projected-internal, Fourier,random,A17 endpoint current rank16. FullGN is unrestricted-current GN proposal plus the same feasibility/Armijo/full-gradient safeguard; receiver/internal are SOM-like/TSOM-style proxies, not native algorithms.
- A17 beats receiver on3/6 final material NRMSE values; median A17/receiver=1.03277967. It is not generally superior. Digit3/E CPU results already show this boundary before later Windows failures.
- Full-gradient controls: digit3 0.37321290, torus 0.46243436; A17 0.55578240/0.67596535. Grad controls each pay86 full factorizations and1248 RHS; do not infer lower total cost from RHS alone.
- All main methods can use the common full-gradient safeguard; raw fallback counts are reported. These are hybrid pipelines, not18 pure ROM/GN updates.
- Controlled wall 8700.103922s, full factorization admissions3084, complex full-solver RHS63808. Mixed hardware/backends; no matched end-to-end speed claim.
- One native access violation and one unexplained helper np namespace exception are preserved. Three trajectories continued from committed prefixes with unchanged rules, all cost retained. Root cause and numerical impact are NOT resolved; do not assert a logging-only issue or new GPU certification.

Please evaluate final-image evidence separately from the frozen-paper full-GN-gap mechanism. In particular: explain why Q>0 before projection does not order nonlinear material truth errors; examine strong controls and gradient fallback; avoid treating thresholded geometry as quantitative complex-material accuracy. Public source is copied without executable regex redaction, and no additional physical rerun/check was performed for publication.

Source: source/Gaussian/A17/SHAPES_3D_BASELINES_20261030/run_comparison.py, original helpers/model dependencies in the same relative layout; collected counter fields and plotting sources are included. REPRODUCTION.md explains replay scope. Complete raw evidence download is listed in START_HERE.md after publication.
