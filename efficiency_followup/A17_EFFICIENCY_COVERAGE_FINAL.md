# 已授权覆盖与真实终态

| 类别 | 授权/实际 | 来源与含义 |
|---|---:|---|
| 专用exclusive profile | 2/2 | 两对象middle/k8，不当生产重复计时 |
| V2几何cache/block控制 | 2/2 | 实际MISS/HIT、同RHS控制；V1另留历史失败 |
| 新冻结policy endpoints | 90/90 | 2对象×3状态×3rank×5policy |
| 原冻结endpoint对照 | 18/18 | 与旧A17相关数组bitwise一致，不增加独立对象 |
| 主matched nonlinear | 24/24 | 2对象×4路线×3轮换计时 |
| P6复用补充 | 4/4 | 2对象×2固定policy，各n=1 |
| top-1/3补充 | 6/6 | 2对象×3计时，CONDITIONAL独立配置 |
| 轨迹接受检查点 | 612/612 | 34×18实际安全材料状态 |
| 缓存整轨迹数组对照 | 6/6 | 1308成员比较bitwise/dtype一致 |
| 费用attempt配对 | 42/42 | 含1次启动失败，未知/活动/未配对0 |

Gate A之后才准入主重建；top-1/3是预声明而后单独准入的有条件补充，不把其执行决定写成看结果前已锁定的主GO路线。

`analysis/derived_v3/analysis_receipt.json`整体PARTIAL保留不改。泛用聚合器期待每个matched分层有三次重复；P6只授权一次、旧历史也一次，故该期待没有满足。对本任务实际授权终态的coverage已经完整，没有等候中的数值队列。availability中的四类实际执行scope均MEASURED。

同时保留“未测”和“未运行”：非线性逐state的完整GN reference是NOT_MEASURED；verification ROM/经验cheap accept/NN是NOT_RUN；生产隐式sync完整归因仍未测量。没有用完成标签替代这些缺口。

独立科学样本为两个固定对象；时间重复、状态、rank和候选没有被当作独立对象。旧第三个near-contact仅参与保存数组诊断，没有被当作新的完整配对重建证据。
