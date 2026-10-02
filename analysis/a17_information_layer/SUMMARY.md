# GN 信息诊断增补：实现与 CPU 验证

状态：PASS（独立有限维代数），80/80 基础测试与16/16 prior/LM整合测试；CPU总时间 0.112s，最大误差基础 2.9754e-14、整合 1.78538e-12，固定阈值1e-9。全部文件只在本目录，不修改selector/父线程运行器，不触GPU/远端。

## 定义与默认尺度

优先按 `A17_GN_INFORMATION_INTEGRATION.md` 执行。默认 prior precision `P=1e-5 I`；`lam` 参数表示solve总曲率（prior+LM），记录 `lm_damping=lam-precision`。

`information_summary(Z, lam_total, precision=1e-5)` 输出：

- `info_trace=2||Z||F²`。
- `info_effective_dim=2Σσ²/(σ²+precision)`；固定prior，不随LM改变。
- `info_effective_rank_prior1/prior10` 用 `σ²/precision>1,10`，非machineepsilon。旧别名 `info_effective_rank_lam/10lam` 同样指prior阈值，以thresholds实际值为准。
- `info_logdet_volume=Σlog(1+σ²/precision)`，等于 real信息的0.5logdet。
- `solve_effective_dim=2Σσ²/(σ²+lam_total)`，单独列，不称物理信息。

重复2谱规律只适用于复杂线性material map Z与共享real坐标 `[Re material; Im material]`。任意Gaussian实切向 T 用 `information_from_tangent(Z,T,lam_total,precision=...)` 或 `information_from_real(J,...)`，不声称重复2。内部诊断J的数据布局为整块all-Re/all-Im；A16实际每照明packing需在入口对residual转换成stacked complex表示。

`info_from_gram(E,normtrace,lam_total, qcomplex=q,precision=1e-5)`：E=ZZ*，normtrace定义为complex ||Z||F²，不能传real trace。利用row Gram的eigenvalues，不形成q×q normal matrix、不做全q SVD。合法负roundoff clipping仅用于Gram PSD数值检查，不用于信息阈值判秩。

## 固定弱空间与任务量

`fixed_weak_space(base_Z,precision=1e-5)` 只存σ²>precision的strong right basis Vs。所有弱方向（σ²≤precision，包括全部nullspace）由 `Pweak=I−VsVs*` 表示，不构造完整material d²矩阵。weak basis固定于base operator hash和prior；LM变化不重新定义weak空间。

`candidate_information(base_Z,child_Z,lam_total,precision=...,weak=...,gnstep=...,residual_complex=...,spectrum=False)`：

- trace_change exact从两端Frobenius norm得到；不需spectrum。
- weak_gain=`trace(Pweak ΔI)`，以totaltrace−strongtrace求出，保持signed值。
- taskweak=`pweak.T ΔI pweak`，名称“approximate model GNstep alignment”；保留完整off-diagonal耦合。
- 它不同于文档中的 `Σ|vi*p|²δi`；退化nullspace里后者依赖basis。测试见同物理nullspace：diagonal sum在原basis为2、旋转basis为4，而invariant quadratic始终4。
- `delta_b=D_real.T r_real`完整实材料向量，缺residual时null+reason。谱不能单独决定step。complex残差通过 `delta_b(baseZ,childZ,residual_complex)` 输入，输出realpacked材料。
- spectrum开启才计算Δdeff与Δlogdet。未计算均为null+missing reason，不能0填充。

API将 weak-task/weak gain标为base ROM诊断，不标fullGNdefect、不标TRUE_WEAK。无fullreference不足以证明fullphysics weak。

## 小Gaussian与大voxel接口

`information_fidelity_real(Jmodel,Jreference,precision=1e-5)` 供小material evaluator：完整reference的H_ref=I_ref+P下，计算信息差的归一化spectral/Frobenius norm，独立密集对照已验证。此函数确实形成small p²矩阵并付Cholesky费用，不能当免费selectorfeature。真实frozen J的后处理由父线程接入；本目录测试是独立小矩阵，不是新Maxwell结果。

`common_material_probes(d,object_id,precision=...,count=16,seed=20261002)` 根据object SHA256生成固定公共实材料基，V.T P V=I。记录objecthash、seed、probehash、实际dimension，和gain/truth无关。`projected_information(JV,metadata=...,full_tangent_rhs=...,full_adjoint_rhs=...)` 只输出PROJECTED_ONLY指标，explicit收费。默认16 public probes不证明完整谱/完整weakspace。该接口没有调用任何fullphysics；父线程必须实际获取并记所有illumination RHS。

## 成本与覆盖

| Quantity | 费用/有效范围 | 缺失规则 |
|---|---|---|
| info_trace / candidate Δtrace | 已有Z后的exact代数收缩，O(rows*q) | 没Z不可反算 |
| deff/r_eff/logdet/spectrum | row Gram形成+小eigh，非免费 | 未做spectrum=null |
| weak trace/taskweak | 固定base strongbasis + child×Vs和实际step action | 无weak/step=null |
| Δb | 已有child/base Z与共同complex residual的adjoint contraction | 无residual=null |
| full information fidelity | 小real material完整J_ref，evaluatoronly | 无reference=null，不代用ROM |
| projected metrics | 16固定公共材料probes；fullactions按illumination收费 | 完整空间指标保持null |

推荐所有candidate仅Δtrace；使用 `fixed_diagnostic_ids(incoming_ids,limit=12)` 在观察gain前锁定incoming顺序，仅前12诊断 + final endpoints计算spectrum/weak细项。`coverage` 记录denominator、IDs与遗漏，不能将12候选coverage写成全池。

信息诊断不改selectorlogic、不产生acceptance/stationarity证书，不引入nuisance、prior训练或Flow。P仅是原任务固定prior precision，不是新的先验模型。

## 测试与保留失败

基础测试核对：实际realified J直接SVD/Gram，与complex重复2、trace/deff、固定weakprojector、weaktrace/task、压缩Gram、general Gaussian tangent、currentphase/permutation、material/data permutation对照。整合16 seeds核对：prior/LM分离、logdet、Δb、信息+RHS的步差恒等式、完整smallreference fidelity、公共probe P正交/可复现/计费、单位与prior一致重标定。

初轮边界witness先sqrt(10)再平方产生10.000000000000002，因此strict >10把浮点值归强；测试原本想检查exact10。保留 `TEST_INITIAL_BOUNDARY_FAILURE.json`。修正为exact Gram diag(1,10,11)独立边界fixture，未修改threshold或分类规则。hard rank本来在临界值敏感，连续deff/logdet仍并报。

## 复现

```sh
Gaussian/.venv_nn/bin/python -B research/delegated/a17_information_layer/test_information_diagnostics.py
```

`TEST_RESULTS.json`保存全部errors、invariance、damping、省略coverage和nullspacewitness。父线程需要接入实际frozenstate、whitening/materialmetric/prior/LM元数据、reference/probeactions与费用；本worker未改父线程文件。
