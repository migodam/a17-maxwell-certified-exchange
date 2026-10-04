# A17 的额外时间花在哪里？

## 结论与计时范围

**对本次冻结的实现，重复构建参考降阶空间（anchor）、候选公共空间和工作区，比完整定向核验（full directional verification）更值得优化。** 原始定向核验 `J_F d` 已共享当前状态的 CUDA FP64 LU；不能把线性系统右端（RHS）数量直接当成主要实际耗时瓶颈。材料外迭代（outer iteration）改变材料状态；同一外迭代内的候选比较则使用固定的线性化。

本报告使用两类互补证据：原完整重建的同步阶段计时，以及两个规定对象 middle/k=8 的独立、可相减的 profiling。前者回答端到端增量，后者定位公共空间内部的重复运算。二者的启动、存档及仪器开销不同，不混作同一次运行。

## 原完整重建：18 次 outer iteration

材料空间分别为平滑 voxel 和壳体 Gaussian。每次迭代求解相同物理模型，接受的 line-search 状态沿用到下一次线性化，因此多数全波分解在 line search 内发生。

| 项目，秒/outer iteration | 平滑 receiver | 平滑 exchange | 壳体 receiver | 壳体 exchange |
|---|---:|---:|---:|---:|
| 初始/沿用 full state 阶段 | 0.223 | 0.208 | 0.213 | 0.214 |
| full gradient / adjoint | 0.081 | 0.084 | 0.080 | 0.092 |
| receiver 或交换公共空间建设 | 0.266 | 11.595 | 0.199 | 10.995 |
| online exchange，含下面的 Jx/Jd | 0 | 2.194 | 0 | 1.472 |
| 约束投影与 fallback | 0.001 | 0.001 | 0.053 | 0.041 |
| full Armijo line search | 10.732 | 11.880 | 21.139 | 16.894 |
| 其余实际费用，未强行分摊 | 0.722 | 6.253 | 0.866 | 3.198 |
| **完整 child wall / 18** | **12.025** | **32.216** | **22.550** | **32.905** |
| Jx 初始化，已含于 online exchange | — | 0.114 | — | 0.060 |
| finalist Jd，已含于 online exchange | — | 0.241 | — | 0.395 |

最后两行是组成项，**不能再加到合计**。未分摊部分是 child 计时范围内未进入 outer phase 的费用，包括输入与物理 setup、阶段间存档、状态写入、垃圾回收和 final metrics。计时开始前的进程启动、导入与守卫，以及计时结束后的最终发布，只由外部 occupation 覆盖。缺少独立计时的部分保留缺口，不按 RHS 猜时间，也不把整个差额归因于 I/O。

原总时间分别为 216.456→579.892 s、405.904→592.292 s。净增量为363.437 s、186.388 s。finalist Jd 总计4.330 s、7.117 s，只对应净增量的 **1.19% 与3.82%**。公共空间建设的增量为203.925 s、194.333 s，分别对应56.1%与104.3%；壳体超过100%是因为 exchange 的 line search 比 receiver 少花76.407 s，存在负的成本差项。

这不是说所有未来实现中 Jd 都便宜，而是说明这个已冻结实现中，“多付了六百多个切向 RHS”与“因此主要慢在切向核验”不是同一判断。

## 新的 middle/k=8 exclusive profile

通过可恢复的函数观察包装器记录嵌套区间；父区间扣除子区间，只累加 exclusive 值。FP64数值算法、候选和物理作用不变。仪器增加同步，因此下表用于归因，不作为无仪器的运行速度。

| 同一状态阶段，秒 | 平滑 voxel | 壳体 Gaussian |
|---|---:|---:|
| 公共空间总体 | 12.732 | 10.882 |
| 原 geometry/internal bank，包含在公共空间内 | 5.486 | 4.437 |
| online exchange | 2.331 | 1.419 |
| Jx，包含在 online 内 | 0.041 | 0.041 |
| finalist Jd，包含在 online 内 | 0.259 | 0.256 |
| driver 实际 wall | 62.078 | 20.493 |
| 已归属 exclusive wall | 29.581 | 18.849 |
| 未归属实际 wall | 32.498 | 1.644 |

平滑对象本次未归属时间较大，现有区间记录不足以进一步归因，不能把62.078/20.493理解成两个材料表示的通用性能比。receiver 在该 profile 先求解，已付费的接收谱可由 exchange 共用；本 profile 也不构成两条独立冷启动策略的速度比。

## A–L 分解的可观测程度

| 所要求的部分 | 实际观察方式与边界 |
|---|---|
| A full physical state / factorization | 原 driver 状态及 line-search 独立区间；LU次数另计。分解主要在表内J项，即line search，不能重复加到 A |
| B receiver backbone | 新 profile 单独观测 receiver modes；原 anchor/pool 内的重复调用属嵌套项 |
| C 64-column anchor | 新 profile 的 `anchor_build`；原计时包含 mode、QR及 reduced-model 工作 |
| D candidate pool | 新 profile 的 `pool_build`；与 anchor 共用 geometry bank |
| E reduced endpoint / scoring | 明确的 endpoint 与 anchor-score 区间 |
| F candidate material solve | endpoint 内阶段统计有记录；GPU批求解与返回仍在一个组合阶段，不能伪造独立 exclusive 总时间 |
| G Jx | 同步的单独定向作用计时 |
| H finalist Jd | 同步的单独 block 作用计时 |
| I bookkeeping / Schur / QR | 新 profile 的 QR/SVD、removal core、incoming endpoint 子区间；扣除它们的父区间后计费 |
| J line search | 原完整轨迹每次 full forward 和接受/拒绝均计入 |
| K full gradient | 原完整 adjoint 阶段；这是 g_F(0)，不是候选当前步处的 g_F(x) |
| L upload/download/synchronization | 新 profile 记录显式 solver upload/download，诊断拆分 RHS、求解、current download、观测及 realpack。没有把它们重新加到 full J 的总时间 |

新 profile 的原策略支付1次 Jx、3次两-finalist block，即7个材料方向×6照明=42 tangent RHS；额外的数值等价诊断支付18 RHS并单列。原策略期间没有新增 LU。完整轨迹两策略的 adjoint 同为108 RHS，LU分别为平滑59/65、壳体115/92；这些LU的阶段时间已经含在上述状态/line-search wall中。

同步次数有两种含义：profiling 每个已闭合 span 调用入口和出口同步；平滑923 spans即1846次仪器同步。这不是原部署的同步次数。原后端把最终数组取回CPU，存在 upload/download 与 CPU观测步骤，但原历史记录没有完整的所有隐式同步计数。报告不把缺少的计数填成零。

下面给出所要求的阶段细分。每列对应一次 middle/k=8 线性化；相邻列不是两条完整重建的时间比。表内公共空间的三项与工作区自身开销互不重叠，online 的子项也互不重叠；物理加载与完整梯度各列一次。LU 本体包含在物理加载，无法从保存计时进一步拆出。材料候选求解包含在候选端点批处理中，没有独立计时，不能把整个批处理时间都归给求解。

| 阶段与计时边界，秒 | 平滑 voxel | 壳体 Gaussian |
|---|---:|---:|
| A 输入与当前完整物理状态加载，含分解 | 11.697355 | 5.470312 |
| B receiver 基底及其端点，共同状态已加载 | 2.546454 | 0.863423 |
| C 64 列参考空间构建 | 6.894808 | 5.481864 |
| D 候选池构建 | 1.535337 | 1.497746 |
| 公共映射、投影与工作区准备 | 4.237977 | 3.871716 |
| 公共空间自身未嵌套开销 | 0.064080 | 0.030868 |
| E/F 三轮候选端点批处理，含材料求解 | 1.392281 | 0.721296 |
| E 参考空间评分 | 0.114462 | 0.028900 |
| G 当前材料步的完整 Jx | 0.040161 | 0.040908 |
| H 三轮 finalist Jd，多右端求解 | 0.257617 | 0.254533 |
| I online 顶层 SVD | 0.089395 | 0.082455 |
| I 初始端点计算 | 0.020787 | 0.003995 |
| I 接受判断算术 | 0.003283 | 0.002048 |
| I online 自身未嵌套开销 | 0.413211 | 0.284859 |
| I 交换引擎共同映射准备 | 0.048767 | 0.001708 |
| K 共同完整梯度/伴随 | 0.085093 | 0.084827 |
| J line search | 未在此冻结 profile 运行；见完整重建表 | 同左 |
| L 生产路径所有隐式同步次数 | 未测，不填零 | 未测，不填零 |

候选端点批处理内部，incoming 端点分别占1.132411/0.511552 s，删除共同核心占0.094620/0.056762 s，SVD占0.077856/0.075900 s。它们已经计入E/F行，不再加入总计。另付费的两次方向等价性诊断中，共享LU多右端求解占0.052375/0.041500 s，接收观测占0.077641/0.075627 s，上传占0.001462/0.001426 s，下载占0.001531/0.001484 s；这些是额外18 RHS的组成项，不是原42 RHS核验阶段的可直接相减耗时。

## 工程解释

原 `domain_bank` 在同一几何、背景 Green 算子和 seed 下跨材料状态不变，但状态上下文每次重建，重复进行几次 dense G_D 乘法及 QR/SVD。精确 geometry-only 缓存可复用这部分，仍须重建当前材料 L、B、anchor/public maps 和 endpoint。减少 finalist/accepted exchanges 则降低方向数量和候选求解/路径费用。

因此完整重建采用配对测量分开比较：receiver、原 exchange、仅加精确缓存、精确缓存加 top-1/最多两次交换。不能只凭 bank 的单次7倍加速宣称整体7倍；完整配对闭合后再裁决部署质量和时间。

证据入口：项目根下 `research/delegated/a17_eff_saved_profile/SAVED_RUNTIME_PROFILE.md`；本目录 `research/PROFILE_MEASURED_SUMMARY.json`、两个 `remote_snapshots/profile_*_complete_v1/`。`analysis/derived_v1/` 与 `analysis/derived_v2/` 是历史截面；最终来源与配对覆盖入口为 `analysis/derived_v3/analysis_receipt.json`、`nonlinear_identity_audit.json` 和 `nonlinear_matched_summary.json`，并须结合原始数组审计。所有后续完整重建原始结果保持另一个版本和独立计时范围。
