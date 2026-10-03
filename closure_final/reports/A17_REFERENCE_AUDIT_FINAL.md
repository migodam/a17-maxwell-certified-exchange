# 最近邻文献与原创性边界

结论：本文最合适的贡献单位是 **Maxwell 指定电流修改的材料后果及其同维数条件交换设计**。Schur 补、双侧低秩法方程耦合、二次收益展开、primal/dual ROM、high-fidelity acceptance 和渐进增秩均不能单列为原创性。正文已按这个边界重写。

## 已取得全文与精确支持

|来源|读到的内容|本文如何使用|
|---|---|---|
|Chen, *Computational Methods for Electromagnetic Inverse Scattering*, 2018，printed pp.161–164 / PDF pp.180–183|Sec.6.4.2 同时处理接收与内部传播。式6.70的交集与式6.71的 receiver-minor projection 是两个构造。已有较小到较大模式数的初始化；式6.72是 Fourier 空间近似。|准确说明 TSOM 已考虑内部物理；不把 projection 错称 orthogonal intersection；不宣称首次 progressive/nested rank。|
|Chen–Zhong–Agarwal, *Subspace Methods for Solving Electromagnetic Inverse Scattering Problems*, 2010，PDF p.8 / printed p.414 起|电流确定部分来自散射数据，模糊部分经优化；TSOM 还分析内部场映射。|支持方法背景和 current/material consistency；综述全文不替代原方法的 priority 审计。|
|Xu–Zhong–Song–Chen–Ran, *Multiplicative-Regularized FFT Twofold Subspace-Based Optimization Method*, 2015，作者 proof PDF pp.3–6|式6–11 为接收 minor 投影、内部 major/Fourier 空间及 data/state residual；式12–14 为材料/电流变量和乘性 TV 正则。Fig.6 的 MF=5→6→12 已展示 progressive Fourier counts。|MR 指 multiplicative regularization。渐进增秩和特定例中继续加模式无额外改善均已有先例，不能包装为本文新发现。|
|de Sturler et al., *Nonlinear Parametric Inversion Using Interpolatory Model Reduction*, 2015；arXiv1311.0922v1 pp.6–9|Theorem3.1 在相应可微、可逆和 forward/adjoint 响应空间包含条件下，在参数采样点匹配 output 与 derivatives。Sec.3.3.1 拼接局部空间后做 SVD。|承认 inverse primal/dual derivative-preserving ROM 已有；本文没有声称发明任务相关降阶或导数保持。|
|Kartmann et al., *Adaptive Reduced Basis Trust Region Methods for Parameter Identification Problems*, 2024；arXiv2309.07627v3 Secs.3.1–3.3|3.1 用梯度/搜索方向丰富 parameter space；3.2 的 primal/dual residual 误差界控制目标误差；式3.17–3.20 接受/拒绝、必要时 full-model fallback、radius 更新；随后用新 primal/dual state 更新 basis；3.21 讨论 estimator 成本。|high-fidelity acceptance、适应性 ROM 和 fallback 是已知原则。本文的 $Jd$ 是指定同秩电流替换所生成的有限材料位移复核，不是新的通用 acceptance theorem。|

新取得的 Kartmann 官方 arXiv HTML 字节 SHA256：`5a19f324e23599eec68468a63e390409e5350b4865d49a69a2697427cb5b54e7`。期刊 DOI 为 `10.1007/s44207-024-00002-z`。该 HTML 用章节/公式定位，不伪造 PDF 页码。上述来源及读取定位、字节身份详见 `research/delegated/a17_closure_primary_sources_final/SOURCE_BINDINGS.json`。第三方全文不复制到公共交付。

## 只有题录或仍 OPEN 的原文

- Chen，SOM，IEEE TGRS48(1),42–49,2010，DOI `10.1109/TGRS.2009.2025122`：题录已由作者书目及 scholarly identity 核对；原始 FULL_TEXT **OPEN**。
- Zhong–Chen，Twofold SOM，Inverse Problems25,085003,2009，DOI `10.1088/0266-5611/25/8/085003`：题录已核对；原始 FULL_TEXT **OPEN**。
- Zhong–Chen，FFT-TSOM，IEEE TAP59(3),914–927,2011，DOI `10.1109/TAP.2010.2103027`：题录已核对；原始 FULL_TEXT **OPEN**。
- S-DBIM 的直接原文方法比较仍 **OPEN**。已有书/综述或准确题录不能升级成已完整读过原算法。

本次有界查证中，Twofold 与 FFT 的 exact-title ScholarQA 请求各出现一次429；没有反复请求。检索/网页无法取得全文不是“未发现先例”的证据，不支持历史优先性。

## 最终贡献表述

**可以明确写：** 材料方向经当前总场注入、经 retained-current feedback 传播，再由接收端返回。这个链条对指定 current-space modification 给出可计算的材料导数与步差后果。信息汇总、直接接收输出和当前有限材料收益是不同量。本文保留同一 receiver 起点与实际电流秩，用少数完整定向检查检验具体替换的材料用途，并在固定规则下记录其收益和失败。

**应作为已有代数或解释：** 低秩 Jacobian 变更的 paired support；Schur/normal-equation subtraction；Woodbury；finite quadratic gain；标准 PCG、Krylov、two-level 或 recycling；logdet/effective dimension；generic goal-oriented/adaptive ROM。

**未建立：** 原始 SOM/TSOM/FFT-TSOM 历史上从未得到同构结果；本文超过 native TSOM 或 generic goal-oriented ROM 的整体反演性能；全局最优电流排名；任何全空间 cheap deterministic certificate。

**不应使用：** first-ever、unprecedented、no previous method；“SOM/TSOM忽略内部传播”；“首次使用feedback/adjoint/high-fidelity验收”；“因为原文没找到所以novelty成立”。

## 对正式投稿的意义

论文已有清晰的机制与条件设计结果，也已公开承认上述成熟思想。剩余的原文缺口是正式投稿前最值得继续关闭的文献条件，而非需要重跑现有数值结果。审稿人仍可能认为“电流替换的物理解释 + 已知 ROM 验收”不足以形成独立方法贡献；正文因此把具体 Maxwell 见证、同秩材料更新效用和被锁定规则的外部对象结果放在贡献中心，而不依赖新命题的数量。
