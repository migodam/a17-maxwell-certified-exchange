# A17 → A18：已付费physics-defect与核验效率接口

状态：`SAVED_INTERFACE_EXPORTED_NO_TRAINING`；本次核对状态为 **数组/source checks 与原 output-interval receipt 六项实际文件绑定均通过**。本次只读取现有文件、做数组比对与哈希核验；未启动 physics solver、reduced solver、网络、SSH、worker 或 training。这些接口不构成 NN model、training 或 Gate A/B/C 通过证据。

## 1. 实际样本与固定 split

| 对象 / split | nonlinear k8 行 / cells | frozen 行 / cells | 总 finalist 行 | exact material-state groups | real material dim |
|---|---:|---:|---:|---:|---:|
| 2002 / development | 100 / 18 | 48 / 9 | 148 | 20 | 3456 |
| 2009 / heldout | 106 / 18 | 52 / 9 | 158 | 20 | 54 |

共 306 个唯一 sample_id、54 个 policy/state cells。nonlinear 覆盖 outer 0–17；frozen 的 early/middle/late 对应 outer 0/3/17，各含 k4/8/16。40 个 material-state groups 和 306 行都不能解释为独立验证样本；不同 k、round、cohort、candidate column 可共用 exact chi state。

固定 object2002 为 development、object2009 为 heldout。按 `object + material_fingerprint + state_hash` 分组，同一 state 的所有候选必须同 split，禁止随机拆行。54 条 partition 的 chi-byte SHA-256 已逐条与对应 saved anchor chi 核对；306 条 sample source_sha256 已与真实 source 文件核对。heldout label 不能用于校准常数重调或 online policy 选择。

## 2. Online feature / offline label 分离及实际数组

| 文件 | 实际数值数组 | 通道与用途 |
|---|---|---|
| a18_features.npz | features (306,26)，FP64；26 feature names；10 metadata 字符串列各 (306,) | online-known 数值与身份信息，禁止 candidate true_Jd/gain/error/original_tau/retrospective Qupper0 |
| a18_online_outputs.npz | anchor_z (306,1536)；full_base_residual_paid_Jx (306,1536)，FP64 | anchor 输出、已付费且因果可用的 full base residual |
| a18_labels_offline.npz | numeric_labels (306,15)；true_z (306,1536)；true_gain/original_tau/error_norm/Qupper0_retrospective 各 (306,)，FP64 | offline teacher/evaluation 与 timing scope labels |
| a18_material_2002.npz | d/x 各 (148,3456)，FP64；sample_id (148,)；fingerprint scalar | voxel 实材料坐标 |
| a18_material_2009.npz | d/x 各 (158,54)，FP64；sample_id (158,)；fingerprint scalar | Gaussian 实材料坐标 |
| a18_partition.json | 54 条 cell 记录 | split/group/source-state join |

前三个 NPZ 的 sample_id 顺序完全一致。per-object 文件用 sample_id join；不要套用全表索引。306 行 d/x/u/true_z 已与 source finalists_round 文件的对应 move_index 列逐条比对，完全一致。数值缺失规则为 NPZ NaN、JSON null、CSV 空白；metadata 字符串不属于 26 列 FP64 matrix。

26 个 actual feature names：`object`, `outer`, `k`, `round`, `dim_material`, `anchor_rank`, `workspace_rank`, `d_norm`, `x_norm`, `anchor_z_norm`, `anchor_base_residual_norm`, `full_base_residual_norm_paid_Jx`, `paid_Jx_norm`, `lambda_total`, `prior`, `prior_linear_term`, `prior_curvature_cost`, `Qhat_original`, `Gamma_relative`, `child_condition`, `core_relative`, `child_normal_relative_residual`, `projected_workspace_current_rhs_norm`, `projected_workspace_current_residual_norm`, `projected_workspace_current_residual_ratio`, `direct_endpoint_fallback`。

所有行 anchor_rank=64、workspace_rank=128。`Qhat_original` 使用原 anchor base residual；区间文件 `Q_A_at_full_base` 使用 paid full-base u。这两个 base residual 不同，不能直接互换两个 gain estimate。

## 3. Material gauge 与 paid physics-defect target

两对象 physical n=1728、volume=0.001953125。实材料 vector 顺序为 `[Re(c), Im(c)]`。2002 的 q=1728，physical expansion 为 c/sqrt(volume)。2009 的 q=27，使用 saved material_tangent_Q (1728,27)，physical expansion 为 Q*c。用 sqrt(volume) 对 physical realification 加权，得到 physical norm。不得将 3456/54 维填充后宣称共用一套 gauge。

2002 material fingerprint：`a11e138d5fa77a6af8f594c1f57df00a71b16ba8b569843afae7719bb1187524`。

2009 material fingerprint：`8f41638a421faa7c8ea9a4f49e9e170f02751445c2c3b0c71f87049d1981bc4f`。

本次 2009 `volume*Q.T@Q` 与 identity 最大绝对差为 6.6613381477509392e-15。已有 gauge_checks 的 54 条 fingerprint match 全为 true，最大 physical-expansion relative error 为 1.9964695034160656e-16；对应 source manifest hash checks 已通过。

实际 paired physics-defect target 是 `delta_z = true_z - anchor_z = (J_F-J_anchor)d`，形状 (306,1536)，属于 realified、measured-data-normalized receiver-output space。它描述 paid full directional verification 与 same-state anchor 输出之差，不是材料 d 的误差。

**当前 NPZ 未独立导出 delta_z 字段**；未来消费者只能在 offline label 通道按 sample_id join 两个数组后相减。本次差分范数与 exported error_norm 的最大绝对差为 1.7347234759768071e-18。标量 error_norm 不含 delta_z 的方向信息。

full_base_residual 为 u=r+J_F x；必须先有 paid Jx 初始化以及此前 accepted fully verified exchanges，不能视为免费 primal feature。true_z、true gain/error、原 tau、retrospective Qupper0 全属于 offline。新增 true_z 采集仍须 full Jd 计费，本次仅消费历史已付费结果。

## 4. Timing label 与 fallback 限制

每个 candidate 的 Jd_full_rhs=6，表示一次 full directional Jd 对应六个物理 illumination RHS。306 行只关联 54 个唯一 timing_scope_id；每一 scope 的 Jd_stage_wall_scope_s 完全一致，它测量同一 policy/state 的全部 finalist rounds，并在候选行上重复用于关联。

禁止按 306 行求和、禁止以方向/RHS 数除出 per-candidate seconds，也不能累加 nested timers 后宣称节省。已有 wall label 不能直接监督单候选耗时。

所有 306 行 direct_endpoint_fallback=false。它是 reduced endpoint online-known fallback metadata；不是 hypothetical full verifier fallback label。尚无新 full-verifier fallback 的实测发生次数、wall 或成功率；不能由全 false 推论后续替代 verifier 可靠。完整 fallback 的政策、成本与门槛仍待 owner 冻结及验证。

## 5. Residual / interval / uncertainty 边界

projected_workspace_current_residual 对应 saved Q span 内的 `L*anchor_current-PB*d`。其 workspace 维度为 128，**不是 full 3N current-equation residual**，也不是 Maxwell output-error certificate；真正 full current residual 未保存。conditioning fields 来自原 finalist move records，不能按 rank 猜测。

g_F(x) 未保存；新增 dense full gradient 仍需 paid full adjoint。retrospective Qupper0=Q_true+0.5||true_z||² 使用 offline true_Jd label，不能作为免费 analytic screen。

后续 output-interval diagnostic 使用 paid u、anchor_z、saved prior terms，无需新 g_F(x)。因此早期 `NOT_RUN_MISSING_GX` 限制只适用于旧路径；已完成的 saved output-interval 仍不等于 deployment，也没有重建改变后的 causal path。

固定 empirical constants：c_anchor=0.9960683840733512，c_displacement=0.0898323733945177，来自 development 的最大 error/norm 比率 ×1.1；heldout 未重调。output_interval_screen_features 有 612 行，即 306 finalist × 两种 radius，并非 612 个独立候选。该文件无 label_*，threshold 固定 0，cheap_accept_allowed 全 false。

| split / object | anchor interval coverage / rejects | displacement interval coverage / rejects |
|---|---:|---:|
| development / 2002 | 148/148 / 0 | 148/148 / 1 |
| heldout / 2009 | 158/158 / 0 | 158/158 / 0 |

全部 offline interval coverage 为 true；raw-positive 与 original-significant false rejection 均为零。唯一 rejection 属于 development frozen cell；两对象 nonlinear rejection 均零。那一次对应 counterfactual 六个物理 RHS，realized savings 仍为零，不能当作实测效率提升。

development coverage 由最大比率拟合加 margin 产生；heldout coverage 只覆盖 saved finalist selections。选择偏差、state/round 相关性、未测对象与 policy path 改变尚未解决。这些 empirical bounds 不是 certified，不允许 cheap approximate acceptance。

## 6. 正确 receipt 文件边界与可复核来源

output_interval_saved.py 第 39 行明确写出 online_features_sha256=sha(a18_features.json)，offline_labels_sha256=sha(a18_labels_offline.json)。这两个字段实际绑定 JSON；online_outputs_sha256 绑定 NPZ。按 writer code 指定的真实路径核对后，原 receipt 六项 SHA-256 均与当前文件一致。

Reviewer note（本次核对更正，不是实验失败）：核对者曾误把 feature/label 的两个 receipt 字段与同名 NPZ 比较，产生两项表面 mismatch；该路径选择错误已纠正。原 receipt 实际绑定 JSON，未要求绑定 feature/label NPZ，不能把这一边界说成原实验错误或 stale archive。本次另建 CURRENT_EXPORT_MANIFEST.json，显式绑定当前接口 NPZ/JSON 及核验结果；原文件未改动。

本次另核对 feature NPZ 的全部 26 数值列和 10 metadata 列与 JSON 一致；label NPZ 的全部 numeric_labels、named scalar arrays、sample_id 和 timing_scope_id 与 JSON 一致。source true gain 由原 u/z/ell/x/d/lambda 计算，其最大绝对差见 EVIDENCE.json；tau、Qhat、scope Jd wall 与对应原文件精确一致。没有运行 reduced solve，也未重新构造 anchor_z。

mtime 显示接口 NPZ 为 2026-10-04 01:24:34+08:00、output_interval_receipt 为 01:28:25+08:00；writer timestamps 与代码读取边界持久化于 EVIDENCE.json。时间戳只提供文件顺序线索，不能证明执行事件或推断重导出原因。当前 correct-path 六项 binding 闭合，不需虚构 archive mismatch 原因。

两个 source manifest 分别含 348 和 476 条 bindings，本次逐条核验 SHA-256，mismatch 均零。其与 output-interval receipt 属于不同核验范围。

| current interface artifact | SHA-256 |
|---|---|
| `a18_features.npz` | `062563aec65c7c754bf3f5eb05a99cbb38e44d38fe3e64968a30d748b152a909` |
| `a18_online_outputs.npz` | `bb88c32cfcf3d066a9b284e3eafc44f96af7280610b4d2a18a1ddd1586d1e33a` |
| `a18_labels_offline.npz` | `fa8ff246825a344e6bc54dd40fb470e505c1c95e1228cecaae925e105f9421a1` |
| `a18_material_2002.npz` | `40087000f27aec992b0ffa7a24b74f38442121aa6e34a4ef347aca8cf2ddd24e` |
| `a18_material_2009.npz` | `560f0335daf3d3c45d9bce878bdc806be5518318553593133a0969f195b723f5` |
| `a18_partition.json` | `47b71fa387a4bf1056040d6ad694f9e80c3b10a2c1082f644fe5f41a7a295d0f` |

EVIDENCE.json 持久化当前 arrays shape/dtype、row/group/cell counts、source-column checks、54 state-hash checks、timing scope checks、两个 manifest checks、receipt 六项逐条结果、writer-code 路径更正与输入哈希。CURRENT_EXPORT_MANIFEST.json 对当前导出另行字节绑定。接口已按来源数组、计量范围及online/offline边界复核；科学裁决以最终效率报告为准。

## 7. A18是否值得主要学习这个defect？

它是一个明确、物理一致的目标，但当前原实现的Jd只占净额外wall的约1.2%/3.8%。消除全部Jd也不能消除重复anchor/公共工作区、存档或line-search成本。应分别核算学习defect能减少的独立切向方向、模型获取/推断/fallback费用和完整wall，而不是以高维输出缺陷本身作为加速依据。其他matrix-free或没有预付LU的场景可能成本不同，本任务没有测试。

object/outer等身份字段用于join和分组，不宜被当成可迁移的物理predictor；40个材料状态组和54个cells不是54个独立对象。本导出只覆盖两个对象的历史已选择finalists，不能据此认定一般候选的误差分布。

本目录 `A18_INTERFACE/EXPORT_MANIFEST.json` 绑定重新交付的字节副本；特征和离线teacher文件分别存放，原文件、JSON receipt和原哈希均保留。可直接消费，但不能将离线true_z或Q混入deployed input。
