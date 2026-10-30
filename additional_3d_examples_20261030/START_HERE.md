# 六类三维目标：A17材料重建完整对照

**36条主轨迹＋2条普通梯度控制已结束。全部18次接受更新，均为MAX_UPDATES，未宣称收敛。**

A17对receiver：3/6材料误差更低，对象比值中位数1.0328。强对照显示它并非普遍更好的成像器。普通梯度在两个固定追加对象上均有更低NRMSE。

## 最终复材料NRMSE（越低越好）

| 目标 | 完整GN提案* | Receiver / SOM-like | 内部传播proxy | Fourier | 随机空间 | A17交换 |
|---|---|---|---|---|---|---|
| 立体数字3 | 52.82% | 70.59% | 41.92% | **41.38%** | 46.33% | 55.58% |
| 字母E | 59.93% | 55.67% | 60.44% | **43.87%** | 55.78% | 68.85% |
| 单环 | 71.06% | 69.61% | **49.89%** | 51.94% | 56.51% | 67.60% |
| 双环 | **53.33%** | 81.83% | 54.29% | 56.44% | 56.17% | 89.57% |
| 三轴交叉杆 | 126.23% | 69.99% | 67.02% | 66.00% | 64.70% | **64.46%** |
| 偏心核壳 | 52.76% | 56.06% | 47.82% | **37.53%** | 57.00% | 81.34% |


*五个电流ROM为k=16；完整GN不限k。Receiver是SOM-like电流空间对照，内部传播是TSOM-style proxy，均非原生完整算法。所有主方法都含同一完整梯度保护。粗体只标主表每行最低值。*

## 阅读与图像

- [中文研究报告：结果、强对照及论文价值判断](RESEARCH_REPORT_ZH.md)
- [三维终态全图](figures/FINAL_SUPPORT_6x7.png) · [原始Re切片](figures/RAW_REAL_XY_central.png) · [原始Im切片](figures/RAW_IMAG_XY_central.png)
- [普通梯度控制的实际切片](figures/NATIVE_XY_MAIN_AND_FULL_GRADIENT_CONTROL.png)
- [逐方法CSV](data/analysis/FINAL_METRICS.csv) · [控制CSV](data/analysis/FULL_GRADIENT_CONTROL.csv)
- [方法含义](METHODS_AND_INTERPRETATION_ZH.md) · [完整费用与故障](COST_AND_FAILURES.md) · [源码与重画](REPRODUCTION.md)
- [给网页版GPT的审阅入口](GPT_REVIEW_CONTEXT.md)

所有图来自保存的实际材料数组；SYNTHETIC TRUTH第一列是合成目标，不能当作重建。统一阈值外形会隐藏背景误差，须一起看原始切片和完整NRMSE。

累计受控计算8700.104秒，full分解准入3084次、复full-solver RHS 63808列。包含失败与接续，未训练网络。2小时上限经用户明确批准扩为3小时，累计费用未重置。

本补充研究使用12³同网格和单个1%噪声实现，且经历未闭合根因的运行异常；不替换原论文的冻结主证据。完整结果与失败一起公开，原稿不变。

## 完整数据下载

[原始检查点／交换轨迹与源码压缩包](https://github.com/migodam/a17-maxwell-certified-exchange/releases/tag/a17-six-shapes-3d-2026-10-30)。公开范围见PUBLIC_SCOPE.md；无需下载大归档即可先看当前报告和compact data。
