# 用户新增 GN information 诊断：执行前补充

来源：A17_GN_Information_Theory_Experiment_Design.md（2026-10-02 用户直接加入）。加入时尚未启动任何远端 A17 Maxwell 作业。A17_RUN_CONFIG.json 的原交换规则保持冻结；以下只增加独立诊断与日志，不用新信息改变选择策略或 gate。

材料空间已按物理范数正交，测量使用原白化，nuisance projection=0。所有照明共享材料变量。对 complex compressed Jacobian Z，实材料 Jacobian 的非零奇异值是 Z 的奇异值重复两次。数据 information I=J^T J 与 GN Hessian H=I+lambda I 分开。

每个 receiver/最终 current endpoint 保存 nonzero information spectrum、trace、effective information dimension、按 lambda 与 10lambda 定义的 effective ranks、绝对 full-GN risk。所有 teacher 候选保存 information trace change；有效谱与 weak-space 诊断采用固定 family-balanced 12 incoming 子池及已接受/最优端点，明确覆盖率与诊断费用，避免为全部候选做昂贵谱分解。

weak-space 规则预先固定为 information eigenvalue <=lambda，包含精确/数值零空间。重复特征值按子空间处理，不取零空间中任意四个 eigenvectors。采用 basis-invariant 的 weak trace change 及投影 GN step 的 directional information change；后者明确区别于文档中对任意退化特征基加权的对角和，不把 reduced model 的初始 GN step 叫作完整目标 residual defect。

优先复用保存的 A16 selected spaces，补 k=4/8/12/16/24/32 的 information–absolute-risk 对照；旧材料步和风险不覆盖。无法绑定到同一 state/gauge 或公共空间的基底记录 UNRESOLVED，不重跑完整 A16。此诊断仍占用原 12 GPU 小时预算。

相关性和排序反转只能说明三层对象的非等价与关联，不能把 information 饱和写成 high-k 退化的已证明原因。原 full objective utility 决定交换；局部 GN fidelity、一次 unconstrained truth diagnostic、最终 nonlinear reconstruction 三层保持分开。没有输出误差证书时不升级安全/停止 claim。

本补充不启动 nuisance experiment、Bayesian inference、Flow、网络或新重建。网络仍须原 teacher gate；新增信息特征不改变留出对象划分。

## 第二份整合文件到达后的尺度澄清（尚无 A17 GPU 结果）

A17_GN_INFORMATION_INTEGRATION.md 指定物理 prior 与 LM 分开。原文字中的按 lambda 信息阈值在此更正：信息指标一律使用旧 resolved config 的固定 P=1e-5 I；LM 只进入实际 step 的 lambda_total。effective rank 阈值为信息/prior>1,10，weak-space 为信息<=prior。另报 solve effective dimension，不称测量信息。选择/接受和原 GN 目标不变。原补充保留以显示形成顺序。

 compressed Z 的行是每个模型的独立观测 QR 坐标，不能与 physical residual 直接配对求 delta_b。delta_b 必须使用真实 reduced model adjoint difference (J_child^T-J_base^T) r；信息谱和 weak 投影只用其相同材料坐标的 Gram。Gaussian full reference fidelity 为离线评价；voxel 16固定public material probes只支持PROJECTED_ONLY结论，付全照明切向费用。数据白化是原归一化尺度，未校准为实际 noise covariance，信息量不解释为 Bayes mutual information。
