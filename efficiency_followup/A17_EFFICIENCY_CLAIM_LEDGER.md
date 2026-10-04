# 效率后续研究的claim账本

原A17科学结果不被本账本替换。这里仅裁决效率任务：完整有限收益与同物理材料更新目标固定，无新ranking/solver/NN理论。

| Claim | 状态 | 原始支持与允许表述 | 边界 |
|---|---|---|---|
| Q_upper0是Q的上界 | proved | 同一g_F(x)，因为\|Jd\|²≥0 | 不证明g_F(x)在线免费；数值门槛不是输出误差certificate |
| 本pipeline已有免费g_F(x) | failed | 共同gradient是g_F(0)，核验base为非零x | 获取精确g_F(x)需额外伴随，部署筛选不推荐 |
| 原Jd主导wall增量 | failed | 4.330/7.117秒，仅净增量1.19%/3.82% | 不把方向数当wall占比 |
| 原Maxwell tangent是真multi-RHS | numerically supported | CUDA complex128共享LU、候选×照明块输入 | 已有实现，不是新贡献 |
| 新block比两列调用更快 | failed | 配对3次控制约0.084vs0.076秒，同12RHS | 控制小且不外推所有规模 |
| 几何cache V2数值等价 | numerically supported | MISS/HIT、两物理对象V/meta字节相同；6轨迹对照1308成员比较相同 | 只缓存几何信息；材料L/B/Jd不缓存 |
| 缓存整轨迹节约约9% | numerically supported | 相同轨迹三次计时中位数 | 两对象、同后端；不是7倍端到端 |
| 位移秩压缩可显著省方向 | failed in tested pilot | 同状态矩阵基本满秩，middle/k8均6/6 | 不否定其他问题中的低秩；未生产部署 |
| top-1/2保住≥90%冻结总收益且≤2方向 | numerically supported | 98.607%/99.257%，2.000/1.889 | 对象聚合，部分单元低保留/不动作 |
| top-1/2保住非线性质量 | failed | 平滑材料+5.95%、留出+21.14%超预锁容差 | 冻结收益不能替代重建质量 |
| top-1/3+cache保住两个对象完整重建质量 | numerically supported | 三次计时、同约束Armijo、材料与留出均低于原 | 补充批次、无噪声、k=8、2对象，无新holdout |
| 推荐路线减少完整方向约一半 | numerically supported | 100→50、106→52；含Jx总RHS708→408、744→420 | 平均仍2.778/2.889，不满足≤2 |
| 推荐路线较原交换快26.89%/9.64% | numerically supported | matched child-wall中位数 | 包含轨迹改变；不全部归因Jd |
| 推荐路线比receiver快/达到统一1.3倍 | failed | 1.958/1.318，Gate C未通过 | 可以说改善原交换部署费用，不能说faster imaging |
| always-reuse或period-3可统一部署 | failed | 前者两对象质量下降；后者平滑不通过 | 未拟合更复杂drift policy |
| 经验anchor半径构成确定性certificate | not established | 保存样本306/306覆盖，非线性cheap reject0 | 样本覆盖不是任意方向保证；不做cheap accept |
| verification ROM值得开发 | not established / NOT_RUN | Jd时间很小、前面已近半方向减少；本任务不扩scope | 不把未运行写成数值失败 |
| 新非线性每状态的R_GN更低 | not established | 没有逐状态完整GN reference | 最终truth误差、约束和full residual分别报告 |
| A18 defect接口完备到可开始独立分析 | numerically supported | 306样本、54cell、40stategroup、824source绑定、12export哈希 | 两对象历史数据，不是训练或泛化通过；混合材料维数 |
| 数值与费用终态闭合 | audited | 34完整轨迹、612可行检查点、42paired费用；不可变snapshot | 源身份/数组审计不是连续物理精度认证 |
| 证据包自包含全GPU重放 | failed / explicitly excluded | 42vendor、17旧source、504大workspace仅身份/哈希 | package提供证据和重分析范围 |

最终科学裁决为CONDITIONAL，唯一统一建议top-1/3+精确geometry-only cache。原机制主线、holdout和噪声证据范围不扩张；本任务未改写原论文或公开问题现场。
