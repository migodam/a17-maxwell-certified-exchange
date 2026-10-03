# 冻结规则在额外 Maxwell 对象上的验证

固定的条件交换在所有十个完整对象中降低了对象汇总的完整 GN gap；中位风险比为 **0.492818**，即下降 **50.72%**。另外一个对象只有 early/middle 通过前置求解检查，其六个已审计单元也改善。Late 的 k=4 为执行失败，k=8/16 在状态中断后未运行；三项均保留在 99 个计划单元中，不能写成十一对象完整通过。

这个结果验证的是冻结规则在额外对象上的材料步保真度。该资产库已用于早期宽域电流选择研究，因此它不是全新盲测数据集。没有发现这十一个对象进入所检查的 A17 teacher、网络或对象特定调参记录；历史汇总结果是否影响过规则选择仍不能完全排除。

## 问题、协议与来源

问题是：在相同实际复电流维数下，已冻结的 receiver-centered exchange 能否在原四个对象之外，使约化模型产生的材料步更接近共同完整 GN 步？对照为 unchanged receiver backbone，唯一改变是 tangent-current 空间策略。材料状态、全 forward residual、真实材料注入、照明、物理材料度量、先验线性项与正则/LM 曲率均相同。

运行前列出所有十一项兼容资产，逐对象测试 early/middle/late 与 k=4/8/16，合计 99 个预定单元。对象缺失、材料 gauge 不兼容等来源排除发生在结果之前。对象 2016 late 的 receiver endpoint 没有通过预设真实法方程残差条件，按合同保留失败；未放宽阈值，未换对象，也未补造状态。

规则锁 SHA256：`a214e38954be1438a7b3e5a84cc6ea595d07544ceff652d6fd84995765a00ae5`。对象 manifest SHA256：`7b83634f34f0b00d61e89e72584a865617690f2f4cbd145c29c56140d2a6e288`。完整执行快照为 `remote_snapshots/primary_complete_v1`，其 ZIP SHA256 为 `60e2ff89601be6a1e0ab5a8de13a79ec35f96199a2cdcd5b26ed90b635206641`。

## 指标及统计单位

\[R_{\rm GN}(U)=\Phi_F(s_U)-\Phi_F(s_*)=\tfrac12\|s_U-s_*\|_{H_F}^{2}.\]

先在每个对象内部对其 eligible cells 的绝对 gap 求和，再计算交换/receiver 比值。对象是统计单位；状态、预算和噪声副本均不被算成独立对象。GN gap 衡量 frozen material-step fidelity，不是最终材料真值误差。完整 reference 仅在在线策略保存后打开；它没有进入排名或接受阈值。

| 来源 ID（仅复现） | 几何 / 材料表示 | 审计单元 / 预定 | receiver gap 之和 | exchange gap 之和 | 比值 |
|---|---|---:|---:|---:|---:|
| 2002 | gaussian / voxel | 9/9 | 0.0011343421 | 0.00063953712 | 0.563796 |
| 2003 | gaussian / gaussian | 9/9 | 0.0039933634 | 0.0015398865 | 0.385611 |
| 2004 | gaussian / voxel | 9/9 | 0.0011409551 | 0.0007409285 | 0.649393 |
| 2005 | contact / gaussian | 9/9 | 0.0010812914 | 0.00031709538 | 0.293256 |
| 2006 | contact / voxel | 9/9 | 0.013249474 | 0.0029114538 | 0.219741 |
| 2008 | contact / voxel | 9/9 | 0.00072855286 | 0.00034115002 | 0.468257 |
| 2009 | shell / gaussian | 9/9 | 0.0025164705 | 0.0013019664 | 0.517378 |
| 2010 | shell / voxel | 9/9 | 0.0027805119 | 0.0016282329 | 0.585587 |
| 2011 | shell / gaussian | 9/9 | 0.0027740055 | 0.0015084356 | 0.543775 |
| 2013 | asymmetric / gaussian | 9/9 | 0.00079191898 | 0.00024813417 | 0.313333 |
| 2016 | asymmetric / voxel | 6/9 | 0.00034494026 | 0.00016989126 | 0.492524 |

## 结果与数值分辨率

十个完整对象：10/10 改善，中位比值 0.492818；四分位区间 [0.331402, 0.558791]。按对象重采样 10,000 次的中位数 95% bootstrap 区间为 [0.313333, 0.564681]，seed=20261002。这是固定资产集合的对象级区间，不提供普遍场景分布保证。

96 个审计单元中 95 个改善、1 个在 evaluation-only floor 内相同、0 个恶化；共 216 个 accepted exchanges。所有终点实际秩均与要求一致。无操作提供了拒绝坏候选的机制，但主要正证据是可分辨的绝对 gap 下降幅度，而不只是 nonworse 计数。

receiver gap 范围为 [2.74148e-07, 0.00308923]；交换后为 [8.73194e-08, 0.000835532]。最小 receiver denominator 仍是相应 evaluation floor 的 6.31e+09 倍；最大完整 reference 真相对法方程残差为 9.55e-12。绝对 gap 与 normalized H-step error 在 `analysis/primary_audit_v2/cells.csv` 中逐项保留，图同时显示高 k 的绝对量，避免只看比值。

纳入部分对象已完成的六个单元作敏感性汇总，十一项可用对象比值的中位数为 0.492524，与十完整对象结论相近；这一统计单独标为 partial-object sensitivity，不能替代原 99 单元覆盖。

## 审计修正和保留的失败

旧 collector 要求整个串行批次成功，因最后一个 child 失败，误拒了同批次的 early/middle 两个完整 child。新 collector v2 仅在父作业有已配对 start/end、普通 exit1、所有完成行在唯一末尾失败行之前、实际导入/代码/配置/输入/数组均匹配时接纳这些 child。十项正反合成测试通过。旧 collector、旧汇总及原始数据未覆盖；父作业 100.609s 的全部费用与最后失败仍保留。这个分析接口修正没有改变任何物理求解结果或质量阈值。

## 结论与限度

结果支持一个清楚且范围明确的 design result：receiver 基底上的少量、经完整定向作用复核的交换，能够在冻结规则的额外三维 Maxwell 对象中显著提高材料步保真度。跨四类几何和两种材料表示的效果没有局限于原四对象。

该冻结验证不提供“全新独立盲测已关闭”、99/99 完整覆盖或速度提升的证据。论文将原四对象与额外冻结规则验证分开报告；最终重建的有限结果另见 `A17_NONLINEAR_TRANSFER.md`，不能由本表中的 GN gap 推出。历史暴露、端点失败和额外物理费用均保留。

来源：`analysis/primary_audit_v2/{STATUS.json,ISSUES.json,cells.csv,SOURCE_BINDINGS.json}`、`analysis/OWNER_PRIMARY_STATISTICS_V1.json`、`research/PRIMARY_PHASE_OWNER_GATE_V1.json`。
