# SOM / TSOM baseline 本地只读核对

日期：2026-10-30。范围：A17 SHAPES_3D 当前共同 GN，A9/A12/A16 本地实现，CLOSURE_R1 引用及已有本地全文。不运行物理实验，不改历史代码，不作新颖性裁决。未联网；原始论文缺口保留 OPEN。本报告的公式描述为概括，非逐字引用。

## 结论与最小方案

1. **当前 receiver 已是 receiver-SVD current ROM + 共同材料 GN，不是 native SOM。** 增加一列“SOM”并复用 receiver 会重复同一算法并误标。
2. 本次指定 A9/A12/A16 范围内，**未找到 native SOM/TSOM 的确定电流 + 模糊电流与材料联合/交替优化器**。现有 `tsom` 函数只产生基底，A12 只产生 enrichment candidates，A16 S1/S2 明确写着 `not native`。
3. 最小可公平扩展为：现有 Full material GN、receiver-SVD GN、A17 exchange GN，再加 **projected receiver/internal GN**（A16 S2）及 **fixed low-frequency Fourier GN**（A16 S3a）。统一外层材料模型、投影、Armijo、初始化、数据、停止条件与总成本。它回答“换 current 空间能否改变最终材料误差”，不回答“超过 native SOM/TSOM”。
4. 若用户要求原方法 baseline，应另实现并审查 **native SOM / book-based SOM adaptation**，将其标为独立优化器。不能把接线 S2 或 A9 `tsom_basis` 后通过共同 GN 的结果称为 native TSOM。

## 当前共同 GN 接口

|路径与函数|实际行为|复用边界|
|---|---|---|
|`Gaussian/A17/SHAPES_3D_20261030/run_shapes.py:158 receiver_endpoint(ctx)`|`U=orth(ctx.receiver_modes(k))`；形成 `UᴴLU, SU, UᴴB`，用 `CoreEndpointBatch.evaluate(I)` 得到材料步长|receiver-SVD Galerkin tangent GN；不包含由数据反演的确定电流部分|
|同文件 `run_case:224–325`|全 Maxwell state 创建线性化；材料 prior+LM；用全物理梯度判断下降；共同可行投影、Armijo及fallback；最终 `metrics` 评价材料|新 baseline 仅替换 proposal 的 current 空间；不可偷换外层步长/预算|
|同文件 `normal_reference:167`、`full_material_jacobian.py:assemble_full_material_jacobian`|全材料 J 后求解正规方程|付费参考，保留成本，不向 selector 泄露|
|`Gaussian/A16/CURRENT_SELECTION_R1/code/operators.py:84 ReducedModel`, `:149 step`, `:259 make_problem`|真实 current Galerkin 模型到 reduced material GN normal solve|可作为任意新 `U` 的统一桥接；需检验 `actual_rank` 与物理度量|

当前 CFG 为 inverse n=12、truth n=14、k=16、outer_cap=18；`SHAPES_3D_20261030` 已有三目标仅 sphere/digit8/depth_bridge。新目标/新规模的公平参数由新协议冻结，不能把本报告当作扩大预算的授权。

## 已有实现清单

|实现|具体路径/函数|判定|
|---|---|---|
|Receiver singular modes|`Gaussian/A9/code/a9_engine.py:793 DenseDDA.gs_modes`；`A16/.../operators.py:315 receiver`|`GS` thin SVD 的右/current vectors；缓存及成本已有记录|
|Exact internal right singular modes|`Gaussian/A9/code/a9_engine.py:802 DenseDDA.domain_modes`|对 `Goff` dense SVD，volume 只缩放奇异值；**n_current>4000 直接拒绝**。n=12 时 current dimension=5184，故当前规模不可直接调用；不得为跑通改历史guard|
|A9 TSOM span helper|`Gaussian/A9/code/a9_engine.py:527 tsom_basis`|receiver/internal分配，orth residualization，再按规则补满rank；它不是完整反演器|
|A9 trajectory `tsom`|`Gaussian/A9/code/run_diagnostics.py:12 basis_for`，`:21 tsom`；`run_inverse.py:run`|前半 receiver + 后半 internal 正交残差。trajectory 明确标 `TSOM subspace in common GN, not native TSOM`。注意该路径不是上行 `tsom_basis` 的完整补秩逻辑|
|Approximate internal candidates|`Gaussian/A10/code/cair.py:72 domain_bank`，A12依赖副本 `Gaussian/A12/deps/a10/cair.py`|randomized GD right singular subspace，2 次 power iteration，返回ritz_residual；randomized proxy而非exact internal SVD|
|A12 TSOM-inspired enlargement|`Gaussian/A12/code/experiment_core.py:15 candidates`，`:22 tsom`|domain_bank(count=8) 后投影掉现有ROM；只是候选。`assemble_closeout.py:184`明确不声称native TSOM reproduction|
|A16 S1|`Gaussian/A16/CURRENT_SELECTION_R1/code/selectors_current.py:534 select_current`，`:546 S1`|receiver prefix，原标签 `receiver-SVD SOM-like; not native SOM`|
|A16 S2|同文件`:548–554`|取k//2 receiver，再追加internal正交残差至k；原标签 `book-inspired projected receiver/internal half; not native TSOM`|
|A16 internal backend|`Gaussian/A16/CURRENT_SELECTION_R1/code/operators.py:324 internal`|接 `domain_bank`，计 GD/GD-adjoint products，原标签 `existing randomized GD proxy; not reproduced original TSOM`|
|A16 S3a|`select_current:555`|按固定 l2/xyz/component 顺序追加 Fourier vectors，不以材料真值调排序|

## 本地 primary source 定位与方法差异

现有引用入口：`Gaussian/A17/CLOSURE_R1/A17_REFERENCE_AUDIT_FINAL.md`。其相对 `research/delegated/...` 指向**项目根目录**，不是 CLOSURE_R1 子目录。实际完整索引为 `research/delegated/a17_closure_primary_sources_final/SOURCE_TABLE.md`、`SOURCE_BINDINGS.json`。本次不执行任何哈希检查。

### 作者教材（当前已直接抽取相关页重新阅读）

本地：`Computational Methods for Electromagnetic Inverse Scattering.pdf`。

- printed pp.152–154 / PDF pp.171–173，Eqs.6.64–6.67：SOM 分别优化 ambiguous current coefficients 与材料，step 3.2 有材料更新，step 4 为目标变化/迭代数停止。printed p.154 明确 Eq6.64 不需求解 forward inverse，电流系数和材料各自二次，但联合四次；这与本项目每个外层先精确求 Maxwell state、再优化材料 GN 不同。精确 Eq6.64 内容应由实现者从原页再核对，不由本报告补写。
- printed p.161 / PDF p.180，Eq6.69：receiver minor 可以写为 `(I−VS+VS+ᴴ) beta`，不需完整 minor SVD。
- printed pp.162–163 / PDF pp.181–182，Eqs.6.70–6.71：交集构造与 practical receiver-minor projected internal-major 构造是不同对象；不能把 `(I−P_S)V_D` 冒称精确 orthogonal intersection。
- printed p.163 / PDF p.182：小 M0 reconstruction 用于初始化较大 M0；printed pp.163–164 的 Eq6.72 将内部主空间近似为 Fourier 空间。不能主张 TSOM 忽略内部传播或首次渐进增秩。

### MR-FFT-TSOM 2015 作者校样（当前已直接抽取PDF pp.3–4重新阅读）

本地：`Gaussian/A14/research/delegated/literature/collected/sources/mrfft2015.pdf`。DOI `10.1109/TGRS.2014.2329032`；作者来源 `https://renchengsong.github.io/papers/TGRS2329032.pdf`。作者校样与最终排版一致性 OPEN。

- PDF p.3 Eq5：确定电流 `I_d=Σ(u_jᴴ E_s / sigma_j) v_j`，由散射数据得到。
- 同页 Eqs6–8：`I_a=(I−VS+VS+ᴴ)V_D+ beta`，或投影 low-frequency inverse DFT。每照明有独立 beta，共用材料。
- 同页 Eqs9–11：归一化 data residual + state/current residual 的目标；变量是 beta 和 chi。**不是把GS/GD基底接在材料 GN 上即完成此目标。**
- PDF p.4 Eqs12–14：乘性TV，MR 是 multiplicative regularization；Eq15 为正则化前材料更新。原文算例/FFT表示为2D，直接搬到3D vector DDA还需处理current定义与polarizability，不能声称原论文原样复现。

### 原始三篇

|原文|本地锁定状态|公式页码|
|---|---|---|
|Chen, SOM, TGRS48(1):42–49,2010, DOI `10.1109/TGRS.2009.2025122`|metadata only；CLOSURE 审计标 FULL_TEXT OPEN|**OPEN**|
|Zhong–Chen, TSOM, Inverse Problems25:085003,2009, DOI `10.1088/0266-5611/25/8/085003`|metadata only；FULL_TEXT OPEN|**OPEN**|
|Zhong–Chen, FFT-TSOM, TAP59(3):914–927,2011, DOI `10.1109/TAP.2010.2103027`|metadata only；FULL_TEXT OPEN|**OPEN**|

本地综述 `Gaussian/A14/research/delegated/literature/collected/sources/Chen_Zhong_Agarwal_2010_subspace_survey.pdf` 的 PDF p.8 / printed p.414 定位沿用已锁索引；本轮未重新抽取该页。综述不关闭原始论文 OPEN。

## 最小接线与公平报告建议

**现在可实现且无需联网才能正确命名：** 新runner中形成 `U_receiver=receiver_modes(k)`；`U_projected_internal=orth([receiver_modes(k//2), internal_modes(k−k//2, fixed_seed)])`，逐列去相关/固定补秩；`U_fourier` 用 S3a 固定顺序。三者都走 `ctx.make_model(U).step()` 或当前 `CoreEndpointBatch` 同等桥接，再走当前共同外层。接 S2 时不要为一个静态baseline构造整个anchor/candidate/exchange workspace并误收其成本；直接取所需同一内部候选 backend。结果显示 actual rank、internal Ritz residual、seed、全部实际setup/build/operator/solve costs。

各目标最终材料 primary 至少保存 complex NRMSE、real/imag NRMSE、support IoU、measured/heldout residual、最终 chi 数组与真实切片，并保留 NO_ARMIJO_ACCEPT/MAX_UPDATES/BUDGET_STOP/ERROR。不能用alpha=1局部proposal材料误差代替最终重建。以相同rank为一表、相同实际成本为另一表；完整GN是付费参考而非同rank竞争者。新增目标不得按baseline表现筛选。

**需要联网 primary-source 核对：** 若计划名为 native SOM2010/native TSOM2009/native FFT-TSOM2011，需要先取得对应原始全文并核对 current construction、objective、updates、normalization、停止与边界；由 `scholarqa-research` 做定向核对，保留失败/OPEN。若仅执行上面的明确“共同GN空间对照”，无需为重新命名而把文献缺口硬关掉。

**native adaptation 最小设计要求（尚未实现）：** 用散射数据固定 receiver-major current；优化 receiver-minor coefficients（每照明）与共享材料以降低data+state residual；把原文current与本地DDA dipole/polarizability变量逐式映射，核对实/虚可行域与边界；同一ground-truth data、初值和最终物理评价，独立优化预算及全成本。不得对native方法强行套用当前full-state GN并仍沿用native名称。原始TSOM的保持/修改应在实施前由Codex完成数学审查。
