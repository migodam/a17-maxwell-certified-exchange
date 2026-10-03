# When Should a Receiver Current Space Be Modified?

This is the final evidence and manuscript package for **material utility and conditional current exchange in three-dimensional Maxwell inverse scattering**. The central question is concrete: at the same retained current dimension, can a few checked replacements make the current model's material step more faithful to the full Gauss–Newton update?

The answer is positive in the reported scope. Across ten complete additional Maxwell objects, the median ratio of object-summed full-GN quadratic gaps is **0.4928**, a **50.7% reduction**. These are frozen material-step gaps, not image errors. A further object supplies six of nine planned cells; the missing late-state cells remain explicit. The additional objects had historical research exposure, so the test validates a locked rule on additional objects rather than establishing a wholly new blind dataset.

## Read the paper first

- [English manuscript](manuscript/PAPER_TAP_FINAL.pdf) and [LaTeX](manuscript/PAPER_TAP_FINAL.tex).
- [Technical supplement](manuscript/PAPER_SUPPLEMENT_FINAL.pdf).
- [中文：给陈教授的研究汇报](manuscript/REPORT_TO_XUDONG_FINAL.pdf), written in the researcher's first person.
- [中文：从零理解与完整论文推导](manuscript/READER_GUIDE_FINAL.pdf), including a Chinese manuscript and bilingual terminology.
- [Final Chinese closure report](reports/A17_CLOSURE_REPORT_ZH.md), [claim ledger](reports/A17_FINAL_CLAIM_LEDGER.md), and [submission-readiness assessment](reports/FINAL_SUBMISSION_READINESS.md).

## The physical and algorithmic chain

Material perturbations excite currents through the current total field and the physical polarizability derivative. Internal scattering transfers these excitations; retained currents can return a signal even when a candidate's direct radiation is weak. A specified current-space change therefore alters both the material curvature and the residual pullback.

The method starts from a receiver-derived basis, proposes one-out/one-in replacements using a fixed richer reduced anchor, and checks at most two finalists per round with full directional tangent actions. A positive finite material gain above an online FP64 significance tolerance allows acceptance; otherwise the current space is retained. At most three exchanges are accepted. The endpoint rank remains exactly 4, 8, or 16 shared complex current columns.

The numerical tolerance uses only online quantities. Physical replay of the original 36 cells preserves all 75 accepted actions, endpoint arrays and risks. It is a numerical significance test, not a deterministic Maxwell error certificate.

## Evidence in one view

- Original development objects: 35 improved cells, one equal, out of 36; median object-summed gap ratio **0.3478**.
- Additional-object validation: **96/99** audited cells, 95 improved and one equal; ten complete objects all improve. One failed endpoint and two not-run budgets are retained.
- Limited noise check: **72/72** cells; 71 improved and one no-op. Median object-summed ratios are **0.3307** at 1% and **0.4026** at 3% noise.
- Constrained nonlinear transfer: three complete receiver/exchange pairs. Complex-material error decreases by **10.84%, 0.22%, and 7.16%**. The near-contact real component worsens slightly. A fourth exchange trajectory stops after 17 accepted updates; its separate baseline is not run.
- Complete nonlinear child-wall ratios are **2.68, 1.74, and 1.46**. The method purchases update fidelity with additional physics actions; it is not an acceleration result.

The controlled physical witnesses separate direct radiation from retained feedback, information magnitude from actual utility, singleton effects from pair interaction, and linear scoring from finite-displacement curvature cost. Their domains and controls are stated in the manuscript.

## What this package does and does not reproduce

This is an **evidence/reanalysis package**. It supplies hash-bound state/action arrays, configuration, owned source, figures, audits and failure records. The release evidence volumes restore binary files through their alias manifest. Forty-two hash-identified external modules are required for full GPU physics replay and are not redistributed here. Saved-evidence reanalysis is distinct from rerunning Maxwell physics.

Read [reproducibility](reports/A17_REPRODUCIBILITY_FINAL.md), [full costs](reports/A17_FINAL_COST_AUDIT.md), [source scope](PUBLIC_SCOPE.md), and [review context](GPT_REVIEW_CONTEXT.md). The private transport and process-command originals are excluded; public fee tables are explicit audited derivatives, not a claim of independent access to private monitors.

The historical package at commit `56097653e6a814399581c3759fc3e3e8f0c2284c` remains intact. This directory is an additive closure. Repository naming retains the historical word “certified”; the implemented method and manuscript use **verified conditional exchange**.
