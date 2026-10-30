# 有界纯阅读终稿 review

状态：**1条可执行发现；其余所读数字与方法/范围表达未发现实际问题。**

只读四篇主文与已保存的 `analysis/FINAL_METRICS.csv`、`analysis/FULL_GRADIENT_CONTROL.csv`；对比其显示精度的原记录值。没有运行研究源码、数值实验、测试或benchmark，没有重算NRMSE、比值、中位数、成本总和或比对材料数组。主线程已有科学裁决不在本review重做。

## 可执行发现（1条）

1. **[P2] START_HERE 的两条 CSV 相对链接在 study 根下失效。** `START_HERE.md` 第26行指向 `data/analysis/FINAL_METRICS.csv` 与 `data/analysis/FULL_GRADIENT_CONTROL.csv`，但当前保存文件实际位于 `analysis/FINAL_METRICS.csv`、`analysis/FULL_GRADIENT_CONTROL.csv`，`data/analysis/` 两目标不存在。建议把本地入口改为 `analysis/...`；若发布包有意将数据放到 `data/analysis/`，仅在发布拷贝中改写链接。不能让study根入口宣称有可点CSV却指向不存在的路径。

## 本轮未发现项

- 主表36项复NRMSE、梯度回退次数，以及正文A17分量误差/IoU/测量残差/留出误差的印刷值，与所读CSV原记录在对应显示精度下一致。字母E完整GN/Fourier测量残差约8.71%/13.87%的例子也与原记录一致。没有发现实际数字抄错。未重新计算主线程已判断的3/6、中位比1.0328或相对比例。
- 普通梯度两个对象分别37.32%与46.24%，18次接受更新；两条各86次full分解准入、1248列admitted RHS的引用与控制CSV一致。主文没有把RHS较少升级成总成本更低，也没有宣称匹配walltime优势。
- FullGN被明确限定为完整GN提案＋共同投影/梯度保护/线搜索；receiver和internal保留SOM-like / TSOM-style proxy范围，未冒称native SOM/TSOM完整优化器。所有终态MAX_UPDATES被明确解释为18次上限，未写成已收敛。
- 普通梯度控制的post-hoc性质及不同历史轨迹限制明确；未发现唯一因果归因。GPU原生崩溃及日志命名空间异常的根因/数值影响范围均保留OPEN，没有改称只影响日志。
- 图文明确区分SYNTHETIC TRUTH、重建和阈值外形；指出弱背景/虚部误差与8-voxel内核限制，没有把漂亮外形称为材料完美恢复。本轮未重新渲染图或取代主线程逐图判断。

本review未修改四篇主文。成本全局累计值、故障时间估计及运行根因未重新统计或诊断；此处只确认主文保留了相应成本/故障范围，而非给予新的数值或科学认证。

## 主线程处置

已把本地START_HERE的CSV链接改为analysis/；发布拷贝按data/analysis布局重定位。只修改文档路径，没有新增物理或数值检查。
