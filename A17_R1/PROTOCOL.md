# A17 receiver 基底材料交换：冻结执行合同

这是用户已批准计划的执行版本。原始 A16 源码、结果与公开问题提交保持只读。所有新实验只使用复制并核对哈希的私有算子及保存的冻结状态。

开发对象是近接触 Gaussian（2007）和分层 voxel（2012）。状态为 early/middle/late（第 0/3/17 个更新前），实际共享复电流维数为 4/8/16，共 18 单元。扩展对象固定为平滑 Gaussian（2001）与非对称 voxel（2014）。规则、阈值和来源在新 Maxwell 标签产生前写入 A17_RUN_CONFIG.json；不能按验证结果换对象或调阈值。

问题分为三个层次：receiver 附近是否有可实现收益；廉价 anchor 是否找得到；完整定向检查与搜索成本是否值得支付。teacher 完整枚举声明 dictionary 的 1-swap，online 仅查看按原 family/atom 顺序轮转的 12 个 incoming 候选，每轮最多复核两个 finalist，最多接受三次动作，接受后重新评分。删除逻辑 atom，随后才正交化，不按旋转后的任意 SVD 列改变邻域。

小 Gaussian 用四线程 CPU，voxel 用现有 CUDA complex128 批作用。公共 L/S/B 缓存跨预算共享，但单独 cold 方法费用包含完整 setup。教师标签、完整参考与材料真值隔离；无约束材料步的 Level-1 收益不替代最终重建质量。不得启动新重建队列。

总 GPU 作业占用上限 12 小时，单作业两小时，80% 显存预警、90% 停止，失败与重试付费。预计两对象计算 1–3 小时、全部条件实验 3–6 小时、包括开发和发布总计 6–12 小时。这些是估算，首个 Gaussian/voxel 单元后用真实分阶段时间更新。

新独立测试不继承附件缺失的历史 PASS。核心相对恒等式容差 1e-9，电流块稳定阈值 sigma_min/sigma_max > 1e-10。shared core 不稳定时使用收费的直接端点 fallback，稳定端点不因此判 infeasible。

实化后所有照明共享实材料变量，极化率导数、物理正交材料基、完整 forward residual、ell、lambda 不变。完整 Q 由缓存 Jx 和定向 Jd 复核，不用完整最优步或 full adjoint 当 selector 输入。

若没有经验证的输出误差上界，所有接受标 HIGH_FIDELITY_CHECK，eps_det 为 null；数值容差不冒充确定性证书。shortlist 停止不表示全域局部最优；完整邻域仍有未排除正上界时标 ABSTAIN_UNRESOLVED，接受三次后标 MOVE_BUDGET_EXHAUSTED。

扩展 gate：两个开发对象均有超过数值不确定度的正收益；对象内先求风险和，两个对象风险改善中位数至少 10%；至少一个 k>=8 有收益。NN 另需四对象至少三个正收益及两个对象 k>=8 有收益；未通过记录 NOT_RUN。有限/一阶诊断统一为同候选的 v/d × 线性/二次评价，旧 A16 比值不作为曲率单因子的因果证据。

发布只建新的 a17-maxwell-certified-exchange 仓库。可执行源码不全文正则脱敏；发布副本须通过数值代数和源字节/语义检查，保留缺失、失败、负结果及外部历史认证边界。
