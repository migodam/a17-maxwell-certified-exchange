# 核验次数消融：一次交换过少，两次交换足以保留主要冻结收益

## 比较设计

两个规定对象各取early/middle/late，k=4/8/16，共18个冻结单元。固定原receiver基底、64列anchor、12 incoming规则、conditional排序、FP64、物理材料度量及在线Q门槛，只分别改变finalist数和接受次数。

- original：每轮最多2 finalists，最多3次accepted exchange。
- top-1/3：每轮只核验第一名，最多3次接受。
- top-2/1：最多接受1次，仍可核验2 finalists。
- top-1/1：只核验第一名，最多1次接受。
- top-1/2：只核验第一名，最多2次接受；若第一名不通过则停止该路径。

每次接受后重新计算 conditional proposals。不是从原六个核验中事后删去列；新policy真正重放。full-reference step只进入离线评价，不进入selector。共90个实际新endpoint，原policy18个终点与已存原A17数组逐字节相同。

## 冻结收益与方向数

收益保留率按每对象9个单元的总GN-gap下降求比；分子、分母分别为新策略、原策略的正接受收益总和，并核对其与receiver到终点的风险下降是否相等。它不是逐单元比值的均值，也不是最后成像质量。finalist是进入完整定向核验的候选；top-1/2表示每轮只核验第一名、最多接受两次交换。

| 策略 | 平滑收益保留率 | 平滑平均Jd方向 | 壳体收益保留率 | 壳体平均Jd方向 |
|---|---:|---:|---:|---:|
| original top-2/3 | 100% | 5.333 | 100% | 5.778 |
| top-1/3 | 100.016% | 2.667 | 100.028% | 2.556 |
| top-2/1 | 93.747% | 2.000 | 77.942% | 2.000 |
| top-1/1 | 93.666% | 1.000 | 77.454% | 1.000 |
| **top-1/2** | **98.607%** | **2.000** | **99.257%** | **1.889** |

top-1/2 将 finalist Jd 方向总数从平滑48→18、壳体52→17，减少62.5%及67.3%。计数不含每个state的额外Jx；完整切向作用仍须加上Jx和每方向6照明。小于100%的收益保留损失与方向节省同时报告；超过100%的个别聚合是conditional路径变化，不能外推成top-1排序总是更好。

按在结果前锁定的Gate A（每对象总收益保留≥90%、平均方向≤2），**top-1/2通过**。仅一次exchange在壳体未达到90%，不能推荐为统一低成本版本。top-1/3方向数仍超过2，属于有条件准入（CONDITIONAL），没有通过主GO方向门槛。它的策略在结果前已经列出，但完整重建诊断的启动决定是在观察到top-1/2的平滑对象质量问题之后；该诊断是补充证据，不是新的独立holdout确认。它与每次接受后重算的“条件候选（conditional proposals）”不是同一概念。

## top-2是否必要、第三次是否必要？

第二名多数时候不改变主要收益，但存在真实例外。壳体late/k=16的第一名负收益，而原方法核验第二名取得约1.695×10^-6正收益。top-1/2在该单元不动作，保留率为0，最终GN gap从原8.385×10^-6变为1.008×10^-5。完整逐单元表保留这一结果。

所以“top-2不必要”过强；正确判断是两个pilot对象的聚合收益主要由第一名和前两次动作提供。top-1/2也有82–89%保留率的单元，不能把约99%的对象聚合写成所有state/rank均保留99%。第三次交换在对象总收益中贡献较小，但不是对所有单元严格可省。

## 从冻结收益到完整重建

Gate A只准入配对重建，并未自动通过最终材料或held-out指标。完整重建固定k=8、18次outer cap、同初始化/约束/Armijo/LM。三次计时重复轮换route顺序，同一对象的重复不视为新物理样本。原策略、仅缓存策略与缓存加top-1/2分开测量，最终Gate B/C及唯一部署建议以 `A17_EFFICIENT_VERIFICATION_FINAL.md` 的已闭合运行裁决为准。

证据：`research/FROZEN_OWNER_AGGREGATION_V1.json`、`research/ORIGINAL_CONTROLLER_MAXWELL_EQUIVALENCE_V1.json`、`research/P8_FROZEN_CELL_LIMITS.json`、`configs/POLICY_PRELOCK_V1.json`、`research/TOP1CAP3_CONDITIONAL_AUTHORIZATION_V1.json`；两对象 `remote_snapshots/frozen_*_complete_v1/`。原A17的36单元、holdout、噪声和非线性结论不被这些policy消融替换。
