# InfoLogger 只读桥接

状态：PASS，`test_info_logger.py` 16/16 mock compressed context测试通过；无fullMaxwell调用。源码 `info_logger.py`，不修改R1、父线程selector或远端。

## 入口

```python
from info_logger import InfoLogger
logger = InfoLogger(
    ctx, precision=1e-5,
    fullJ=small_real_full_reference_J,  # optional evaluator only
    V=public_material_probes, fullJV=parent_paid_fullJV,  # optional projected evaluator only
    probe_metadata=metadata, full_tangent_rhs=paid_tangent_rhs,
    metric_hash=material_metric_hash, whitening_scope=whitening_label,
    whitening_hash=whitening_hash, material_basis_hash=material_basis_hash,
)
base = logger.base_summary(ids, U, x, spectrum=True, evaluator=True)
candidate = logger.candidate_summary(child_ids, child_U, child_x,
                                     spectrum=False, weak=False)
final = logger.final_summary(final_ids, final_U, final_x, evaluator=True)
```

`record(ids,U,x,base_model=None, spectrum=False,weak=False,evaluator=False)`为底层入口。已有base_model可直接传入。`model(ids,U)`返回cached reduced model，无reference内容。

- `selector_information`：允许的reduced model信息和候选变化，仅包含compressed Z的material normal信息。
- `evaluator_information`：显式evaluator=True才返回，包含fullreference fidelity/父付projected reference，不得送入proposerfeatures。未提供对应reference则null+reason。
- metadata记录prior precision/hash、LM damping、material metric/basis/whitening hash/scope、actual currentrank/materialdimension、model-specific compressed rows。
- `wall_model_build_s`, `wall_information_s`, `wall_rhs_s`, `wall_evaluator_s`, `wall_total_s`实际记录所有本地动作。调用既有ctx模型会进入既有ledger；本helper自身不调用fullMaxwell。

默认每candidate只做trace与Δb。父线程负责固定family-balanced12 incoming per outgoing，以及final/teacherbest的spectrum/weak开关和coverage denominator。Logger不挑选候选、不改selector、不宣称全邻域certificate。

## 最关键的行坐标边界

`model.normal.Z=stack_p(Rc Fp)`只保留数据范数的经济QR等距表示。不同model的Rc来自不同CV，因此Z的行数、行坐标都可不同。各自的Z*Z在共同材料坐标下可以比较，故trace、Gram谱、weak projector/task quadratic都合法。

**不得**把这种compressed Z与physical `ctx.r`配对求Δb，也不得直接`child_Z-base_Z`当物理D：即使行数恰好相同也不能保证同一行坐标。

Bridge只计算：

```python
delta_b = child_model.jt(ctx.r) - base_model.jt(ctx.r)
```

这通过共同物理白化real data coordinates返回完整shared real material向量。默认以完整list+arrayhash记录；提供output_dir时保存npz+数组hash/文件SHA256/path。candidate的generic compressed-residual接口被显式禁用，其Δb来源标注为bridge physical jt。

weak task使用base_summary缓存的base GNstep（若explicit external base_model则使用caller传入x并标source）。它叫approximate model GNstep alignment，不叫fullGNdefect。

## 缓存和费用

缓存key为原atom IDs tuple + actual U bytehash。相同ids/U重用ctx.make_model与model信息；相同ids但U改变重新build并实际计费。信息summary按spectrum开关缓存，fixed base strong-right basis缓存。Δb仍执行物理jt并记wall，不假装免费。

frozen ctx.r/ell/lam或metadata变动后拒绝旧logger复用；必须新建。外部L/B/S callback变化同样必须由父线程创建fresh frozen logger，helper不能自动证明callback内容不变。

小Gaussian full fidelity走`model.j(I_d)`与提供的fullJ，material normal矩阵small dense evaluator计算。大voxel evaluator仅走父提供固定V/fullJV与`model.j(V)`；PROJECTED_ONLY标签保留。父付fullJV的tangent/adjoint费用作为one fixed setup展示，不能将每条record里的重复view相加重复收费。

## 验证与复现

```sh
Gaussian/.venv_nn/bin/python -B research/delegated/a17_information_layer/test_info_logger.py
```

16 seeds20261002开始：mock重建真实形式CV/F/Rc、per-illumination real data packing；base compressed rows4、child6、共同physical real data rows20。检查Δb直接physical J、trace-only nulls、cachebuild次数、final弱诊断、referencefeature隔离、PROJECTED_ONLY收费、同ids改变U重建，以及LM变化拒绝复用。结果 `BRIDGE_TEST_RESULTS.json`。这些是接口代数验证，不是真Maxwell pilot或runtime收益。
