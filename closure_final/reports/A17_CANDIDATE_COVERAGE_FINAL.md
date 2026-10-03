# 候选覆盖、排序与路径：最终离线封口

**完整字典排序的主要损失很小，但固定12方向短池会遗漏显著机会。** 这是冻结方法的已观察边界，而不是新增选模规则。teacher在原线上决定全部结束后计算，标签不回传到selector。

## 问题与范围

在receiver端点周围枚举固定154方向字典的全部一换一组合。k=4/8/16分别有600/1168/2208个声明动作，无效端点也进入分母。随后最多走三个按完整收益选择、每次重新枚举邻域的贪心动作。它是明确域内的初次1-swap机会与有界局部路径，不是全局oracle。

33状态/99单元中，32状态/96单元来源核验通过且完整。非对称voxel的late状态因先前primary端点失败固定排除；对应3单元仍标NOT_RUN_PRIMARY_INCOMPLETE。没有为了覆盖结果而修复、重算或换对象。

## 先看初次交换机会

每个单元取全字典最大正完整收益Q_*^+；对象内先求和，再计算捕获比例。全字典anchor top-one还经过同一在线数值显著性/no-op判据。零机会不提供权重。十个完整对象的结果为：

|量|最小|中位|最大|
|---|---:|---:|---:|
|全字典anchor top-one保留的初次机会|97.302%|99.538%|99.985%|
|固定12移入方向短池内最好动作覆盖的机会|49.119%|72.951%|95.307%|

所以当前更大的可见损失来自候选取得，而不只是近似排序。这个比例不能解释成整个部署路径保留了99.5%的最终改善；它只度量相同receiver起点的一次决策。非对称voxel对象只有6/9观察，不能填成完整对象统计。

## 再看路径

完整字典贪心路径的终态gap在90/96单元更低；线上冻结路径在5/96单元更低；1单元在evaluation-only floor内相同。这本身说明局部greedy teacher不是全球最优：不同第一步会进入不同条件邻域。终态差距同时含候选覆盖、anchor排序与路径交互，不能全称为score regret。

15条teacher路径以数值局部1-swap停止、81条达到三动作上限。没有实例化合法输出误差半径，因而“数值局部停止”不是deterministic local-optimality certificate。线上短池之外没有完整上界，线上abstention更不能证明全字典没有好动作。

## 来源与复算

- 原始封闭快照：closure_final_closed_v1，receipt SHA256 `0d1463078905fcdc54dcc215d47a34e6a1629629691439805c44effeb966f7e8`。
- 最终来源收集器：collectors_v5，coverage evidence SHA256 `e4f7c13169e570d49d6b5e0e16e8111c9c94e891dd419cded826c2652c40d978`。
- 实际运行配置SHA256 `a214e38954be1438a7b3e5a84cc6ea595d07544ceff652d6fd84995765a00ae5`，未更改。
- owner派生统计：`analysis/OWNER_COVERAGE_STATISTICS_FINAL.json`；独立保存数组复核另列，不把collector结构核验冒充Maxwell重算。
- 完整费用9039.907秒属于offline coverage与campaign总账，不作为线上cheap selector费用或新增独立样本。

## 对论文的影响

保留正面结论：receiver-centered条件交换在相同实际电流维数下改善受测完整GN材料步。进一步明确设计限制：排序准确与机会覆盖不是同一问题，当前候选取得仍是瓶颈。A17不为此开发新网络或新selector，候选取得只作为后续研究问题。

## 独立保存量验算的实际范围

285个完整checkpoint、377720个声明动作、370521个可行score索引与初次capture均独立一致。270个所存选中路径方向的Q复算最大差7.81e-18；96个actual-minus-teacher二次差最大9.76e-19。15个saved-dense-J状态的90个端点绝对gap最大差3.25e-19。

全候选D/Z未全部存储，不能宣称全部标签已由独立物理公式重算；17个无denseJ状态也未独立重算绝对gap。由Js相减恢复Z改变了低位字节，不能回证原输出hash，但所存material direction hash270/270精确。这个边界不影响由完整已存label域重算的捕获率，却限制public arithmetic check的范围。
