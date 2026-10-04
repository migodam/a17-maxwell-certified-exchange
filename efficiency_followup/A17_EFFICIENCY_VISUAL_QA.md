# 科学图视觉检查

主线程已经查看全部11张最终PNG及其数据/分层标记；没有生成示意图代替计算结果。PDF/SVG与同一绘图源和数据绑定。

## 首选5图

1. `analysis/derived_v3/figures_final/01_frozen_gain_vs_Jd`：两对象固定状态收益与实际方向数，原/轻量策略清晰。
2. `analysis/derived_v3/figures_final/03_original_18_update_net_vs_Jd`：实际额外成本与Jd的小占比。
3. `analysis/owner_nonlinear_v2/figures_owner/owner_quality_wall_physics_01`：两个对象分别给时间—complex材料/留出误差，不把重复当独立对象。
4. `analysis/owner_nonlinear_v2/figures_owner/owner_physical_counts_2002_physics_01`：平滑六类物理成本/次数。
5. `analysis/owner_nonlinear_v2/figures_owner/owner_physical_counts_2009_physics_01`：壳体相同指标。

三次组用中位数和范围；P6单次用不同marker。top-1/3补充分支与主批次有明显区分。图注中的current rank始终k=8，计时路线没有降低k。

## 保留但不作为主图的6图

泛用绘图器按driver/config身份分层，还产生3张nonlinear散点图及3张wall ratio图。main组的重复标签局部重叠；P6与top-1/3组自身没有receiver路线，其wall ratio面板因此为空。空面板表示同层基线缺失，不是0、不表示任务未运行。owner图通过已核对的跨driver同physics对照呈现这些路线，不能将泛用脚本的空值补成假数据。

这些诊断图及原HANDOFF文档保留在本地；可移交包采用首选5图。未挑最好重复，未删不利路线，P6负结果仍进入owner图。

## 数据与重画

owner图读取34条已审计trajectory，来源为`analysis/owner_nonlinear_v2/AUDIT.json`，并用`analysis/derived_v3`声明合同。源码`code/plot_owner_efficiency_pareto_v1.py` SHA为`5f30c151672285f9c29c09dc3d472149b7bb3c11857c7be71a3bf2c79b019f95`。

泛用绘图源`code/plot_efficiency_final.py` SHA为`29b3548bab5324551e36b9f8ef561bf69f09841f6a9704a3e449c4ee34b70b58`。个别自动HANDOFF文本字面仍提到derived_v2；本次真实命令和数据是derived_v3，最终复现说明给正确路径，未更改图的数值。

报告中的四舍五入来自未舍入原始表；完整精度见owner summary、JSON/CSV。图像清晰不替代科学gate，最终裁决CONDITIONAL。
