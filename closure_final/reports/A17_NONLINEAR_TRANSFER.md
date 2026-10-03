# 非线性迁移：材料步保真度与最终重建分别验收

预先固定的四个对象中，三组 receiver / verified-exchange 对照完成相同的 18 次外层更新。三组复材料相对误差均下降，幅度分别为 10.84%、0.22%、7.16%；留出测量误差均下降。近接触对象的实部误差略升，复误差几乎中性。第四组未形成完整对照：exchange 在第 18 次更新前（零起始 iteration=17）的 receiver endpoint 残差检查失败，保留 17 次已接受更新；receiver 作业按队列停止规则未运行。这是有限的 secondary reconstruction consequence，不能把三组结果提升为普遍成像改善。

## 为什么要做这个检验

冻结 GN quadratic 比较只回答某一材料状态下的线性化步是否更接近完整步。实际反演还会投影材料约束、缩短步长、改变总场并重新线性化。原先裸 `alpha=1` 的真值诊断不能代表这些操作。本检验使用共同的约束和 full nonlinear objective 的 Armijo line search；两条轨迹只改变 tangent-current policy。

## 固定合同

四个对象按几何与材料表示预选：平滑 voxel、近接触 Gaussian、shell Gaussian、非对称 voxel。来源 ID 仅供复现，分别为 2002、2005、2009、2016。无新增噪声；实际共享复电流秩为 8；初始化为保存的 `0.1+0.04i`。共同 prior 为 `1e-5`，第 t 次线性化的总正则为 `1e-5+0.01*0.3^floor(t/3)`，线性 prior 项也保留。

材料约束为实部对比度不低于 -0.5、虚部不低于 0；使用已有 feasibility projection。非下降方向触发共同 full-gradient fallback。Armijo 常数 `1e-4`，从 alpha=1 逐次减半，最多 24 次；更新上限 18 次，既有 small-step 停止规则从第 9 次后生效。完整 forward、约束、prior、line search、初始化均相同。controller 不取得完整最优材料步、完整 Jacobian 或真值。

冻结驱动 SHA256：`db4fbfc15503408314f948956a07cb5298940aa4a978d074490d8f84d1d73dd6`；derived phase lock：`07bc4efd04a87f88623b123fc1df4beb335c818c8e4a9418a75a544de3ee273d`；原算法 lock `a214e...` 不变。

## 完整终点

以下材料误差均为完整三维物理材料上的相对 L2 误差，不是切片误差。

| 几何 / 材料空间 | 复材料 receiver → exchange | 实部 receiver → exchange | 虚部 receiver → exchange | 留出测量 receiver → exchange |
|---|---:|---:|---:|---:|
| 平滑 / voxel | 0.302660 → 0.269856 | 0.281136 → 0.251563 | 0.743882 → 0.650636 | 0.00663726 → 0.00283650 |
| 近接触 / Gaussian | 0.672191 → 0.670681 | 0.681040 → 0.683457 | 0.443896 → 0.285886 | 0.0105207 → 0.00708006 |
| shell / Gaussian | 0.609640 → 0.565989 | 0.608147 → 0.565060 | 0.697551 → 0.621902 | 0.0141900 → 0.0121793 |

测量残差分别从 0.00663658、0.01053154、0.01417860 降到 0.00283664、0.00708324、0.01217834。带 prior 的实际 nonlinear objective 分别从 `2.28123e-5`、`5.55296e-5`、`1.01132e-4` 降到 `4.82595e-6`、`2.51447e-5`、`7.48145e-5`。两者均完成 18 个接受更新；这不是假定相同最终图像的线性 solver 加速实验。

## 完整费用

| 对象 | receiver child wall / s | exchange child wall / s | wall ratio | full factorization receiver / exchange | full forward RHS receiver / exchange | full adjoint RHS receiver / exchange | extra full tangent RHS exchange |
|---|---:|---:|---:|---:|---:|---:|---:|
| 平滑 / voxel | 216.456 | 579.892 | 2.679 | 59 / 65 | 354 / 390 | 108 / 108 | 708 |
| 近接触 / Gaussian | 407.715 | 710.489 | 1.743 | 116 / 127 | 696 / 762 | 108 / 108 | 744 |
| shell / Gaussian | 405.904 | 592.292 | 1.459 | 115 / 92 | 690 / 552 | 108 / 108 | 744 |

child wall 包含每条轨迹自己的 physical setup、基底获取、全部接受/拒绝与终点评价；不是 warm selection time，也未把 enclosing monitor overhead 再加进去。full tangent RHS 包含每次线性化的 Jx 初始化及 finalist Jd；Jd 复核本身分别为 600、636、636 RHS。两策略的 full outer gradient 费用都保留，验证 finalist 没有新增 full adjoint。每条轨迹各实测一次，因此这些是描述性 all-in 费用，不是有重复计时区间的加速结论。

七次真实 nonlinear attempts 的 occupation 合计 3704.671 s，包含失败作业的 765.593 s。失败费用没有从结果比较或预算中消失。

## 失败与表示边界

非对称 voxel 的 exchange 在 iteration=17 遇到 `RECEIVER_ENDPOINT_NORMAL_RESIDUAL_FAILED`：它为开始当次交换构建的 receiver 起始端点没有通过验收，因此 exchange 轨迹在第18次更新前停止；17 次已完成、可行且通过 line search 的更新均保存。这不是独立 receiver-only 轨迹的运行失败，后者根本没有运行。没有降低残差阈值、换精度或重跑。随后 receiver-only 作业为 NOT_RUN；不使用旧轨迹或最后安全状态伪造完整终点。计划分母始终为 4 对象、8 方法作业，完成 6/8、失败 1/8、未运行 1/8。

驱动的 raw `executed_step_norm` 在 Armijo 全拒绝路径会记录下一未检验 alpha 对应的非零提案长度，实际材料状态保持不变。派生审计以接受状态和保存数组计算 actual accepted-update norm，拒绝时为零。当前 125 个已保存更新全部接受，这个报告字段缺陷未触发；冻结源代码没有事后修改。

## 科学解读

结果提供了两层不同强度的证据。局部层面，锁定交换明显降低完整 GN gap；非线性层面，三个完整对象的复材料及留出测量误差改善，但效果随对象变化，近接触的复材料改善只有约 0.22%，实部还略差。较小数据残差不能代替材料恢复质量。现有证据支持把重建作为有限 secondary consequence，论文主线仍是 mechanism + conditional design，不以 improved imaging 或 faster reconstruction 定位。

证据快照：`remote_snapshots/nonlinear_phase_closed_v1`，receipt SHA256 `2ed23fccc79f7a53d1203162c4ff64dd80e20a1b7b1eb55950664768171a99b7`，ZIP SHA256 `7f05f3bf5b588bb9c3a67daf97997c64deea26e495b25ad6a99c5185919c6ad3`。saved-only audit 检查 125 个更新的可行性、Armijo、连续性、终点数组、计数器及 actual import 路径/哈希；没有新增 forward solve。完整统计为 `analysis/OWNER_NONLINEAR_STATISTICS_V1.json`，原始与失败出处均保留。
