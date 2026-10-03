# A17 最终 claim 与证据边界

论文主张是：**在固定实际电流维数下，从接收导出的基底出发，少量条件交换可以显著提高受测 Maxwell 问题中的完整 GN 材料步保真度。** 物理因子解释一次指定改动如何进入材料问题，完整方向收益决定这个改动对当前残差是否有用。表示维数相同，取得候选和核验的费用不相同。

下面的状态严格使用五类：`proved`、`numerically supported`、`holdout validated`、`not established`、`failed`。其中 `holdout validated` 指规则锁定后的额外对象验证；这些对象有历史研究暴露，**不表示已取得全新盲测总体**。成立的既有代数也不自动成为新颖性 claim。

| Claim | 状态 | 条件、证据与解释 |
|---|---|---|
| 材料切向通过当前总场和真实极化率导数进入电流方程，完整导数为 J=SL^{-1}B | proved | 固定可逆的离散 Maxwell 电流系统，共享物理正交实材料变量；DDA 的 a'(χ) 保留。CORE_THEORY、FINAL_DDA_MODEL。 |
| 指定稳定电流增广的导数差等于 Y_C Γ_C^{-1}N_C | proved | 保留块及所用 Schur 块可逆；多照明提升和实材料拉回保持一致。Schur 消元是已知工具。 |
| 接收输出包含保留反馈：bar C=C+R_0 X G_D C | proved | V 与 C 在声明电流度量下正交，R_0 为保留空间解算子；DDA 用相应 a(χ) 与自作用约定。该频域关系不引入时域控制稳定性结论。 |
| 两组材料方向包含指定模型改变的精确步差 | proved | 共同 B,r,Λ,ell、正定正则、无约束精确端点步、未截断非零签名。是支撑包含，不是最小空间或自动低维保证。 |
| 两端 GN 曲率可以均正定而其增量不定 | proved | 交叉项 J_o^T D+D^T J_o 可为负；改变的是已有数据的近似模型。扩大同一完整材料可行空间是另一问题。 |
| 当前材料步由曲率改变和残差相关偏置共同决定 | proved | d=-H_n^{-1}(Δh+ΔI s_o)，固定共同残差与正则；不将负收益单独归因于某一项。 |
| 实际方向的完整收益可以用缓存 Jx 与 Jd 计算 | proved | 完整 frozen quadratic、实际位移 d、正确实材料内积；公式适用于任意 d。二次收益恒等式是已知代数。 |
| 每 finalist 的方向复核不需要新增完整伴随 | numerically supported | 真实 callback 和已保存计数一致；需要每照明一次 Jd，Jx 初始化也收费。不是零全波费用。 |
| 在线接受不再依赖完整最优材料步、真值或 evaluation floor | numerically supported | online_tolerance.py 仅接收在线 r,u,z,x,d,ell,λ；真实物理 replay 的 36 单元、75 接受、空间/步/Js 保持一致。 |
| 原四对象固定 k 的 GN gap 改善 | numerically supported | 35/36 improved，1 equal；四对象先求和，中位比 0.347848，即下降 65.2%。是开发证据。 |
| 冻结规则在额外对象中仍改善 GN fidelity | holdout validated | 十完整对象全部改善，中位比 0.492818，下降 50.72%；96/99 可审计，95 improved/1 equal。另一个对象六单元改善；late 1 FAIL+2 NOT_RUN。历史暴露限制保留。 |
| 上述改善是超过数值 floor 的绝对收益 | holdout validated | receiver gap 2.74e-7 至 3.09e-3；最小分母/floor 为 6.31e9；参考真相对残差至多 9.55e-12。比值同时配绝对 gap。 |
| 有限噪声下仍出现材料步收益 | holdout validated | 四固定中期对象、1%/3%各三份复噪声、三个 k；72/72 可审计，71 improved/1 no-op。对象中位比 0.330653/0.402576，各残差有自己的 reference。 |
| directly-dark 电流可通过保留散射改变材料导数 | numerically supported | 受控 5³/6³/7³ Maxwell 例子：direct norm <1.7e-16，dressed norm 0.0672–0.0888，导数变化 3.56–5.22%。证明可能机制，不因果归因所有已接受动作。 |
| 信息体积增加及一个曲率保真标量改善不足以判断材料用途 | numerically supported | 真实 witness 的信息体积增加 6.3127、有效维数增加 1.8613，但 Q=-5.5754e-5。不是“信息理论无用”。 |
| 材料相关性可为条件性、非加性 | numerically supported | 声明的 540 对中有 5 对 singleton-negative/pair-positive；代表 QA<0、QB<0、QAB>0。增秩见证不是同秩交换陷阱或普遍非次模 theorem。 |
| 实际有限位移的曲率代价影响候选排序 | numerically supported | 同候选 2×2：mandatory top-one、no-op 故意关闭时，actual-d linear 29/36 负，quadratic 1/36 负；不是部署失败率。旧 0.1171 不归因于单独二次项。 |
| 冻结交换可产生有限非线性重建后果 | numerically supported | 三对完整、共同约束/Armijo/18 更新：复误差下降 10.84%、0.22%、7.16%；留出测量下降。接触实部略差；第四对没有成对终点。不能概括为普遍成像优越性。 |
| 全字典anchor可较好排序初次交换机会 | holdout validated | 十完整对象、固定receiver初始邻域，全字典top-one正收益加权捕获97.30–99.99%，中位99.54%；不是部署整条路径捕获率。96/99离线单元完整。 |
| 固定短池的机会覆盖是当前瓶颈 | holdout validated | 同一域下12移入方向最好机会覆盖49.12–95.31%，中位72.95%；局部路径差距不能全归于排序。未新增或重调候选。 |
| 完整字典局部greedy是全局oracle | not established | 线上在5/96单元优于三动作teacher，teacher在90单元更好，1在floor内同等；路径与初次机会分别报告。 |
| 全部原数值作业与失败费用已闭合 | numerically supported | 61已配对attempt、无未知数值进程/锁；历史4889.515s+closure15072.343s+两次preflight0.202s=19962.060s，5.545h；嵌套费用不重复相加。 |
| 原宽域 from-scratch 材料选择普遍优于 receiver | failed | A16：15 已观测对象、258/288 单元、中位对象风险比 1.0947、仅 6 对象改善。条件交换回答更窄问题，没有覆盖旧负结果。 |
| paired material repair 优于等秩材料 Krylov，或获得加速 | failed | 原 smooth/sharp 的 Krylov 均强于 paired；paired paid wall 1.416，RHS 66→342。结构支撑不等于更强 solver。 |
| 非线性全部四对已完成 | failed | 非对称 voxel 的 exchange 轨迹在第18次更新前，为开始当次交换构建的 receiver endpoint 发生 residual FAIL；保留17个接受更新。独立 receiver-only 作业 NOT_RUN，未重试，不能推断未运行基线一定失败。 |
| 更多方向、信息体积、更小 Jacobian error 自动改善材料估计 | not established | 当前理论和真实负收益例子均不支持这种无条件推断。GN gap 与 truth error 分别验收。 |
| 全字典局部最优或全局 current selector | not established | 在线只有12 incoming、每轮最多2 finalist、最多3 accepts；无合格 shortlist 动作报 abstention，不能证明字典没有好方向。 |
| deterministic certified acceptance/optimizer | not established | 尚未实例化合法 Maxwell output-error radii；FP64 significance、solve residual 和 repeatability 不能替代。历史 repo 名不构成成果措辞。 |
| 加速成像或相同计算成本下优越 | not established | 三完整轨迹 child-wall 比 2.68/1.74/1.46；每轨迹实测一次。完整取得/拒绝/失败费用保留。 |
| 原生 SOM/TSOM/FFT-TSOM 被击败 | not established | 比较为 receiver-centered tangent-current design；历史传播候选是 proxy，不是原生联合目标/current optimization 复现。 |
| 首次反馈、首次任务相关 ROM、首次 quadratic acceptance | not established | Schur、Woodbury、Krylov、goal-oriented/primal–dual ROM 与二次评价已有成熟 prior art；原生方法全文缺口继续 OPEN。 |
| 公开材料支持完全自足 GPU rerun | not established | 42个hash-identified外部模块及其兼容内核仍是依赖；公开目标为 evidence/reanalysis package。 |

## 科学定位

**Strong mechanism + conditional-design paper** 是这组证据可支持的定位：材料入口、保留反馈和接收返回提供 Maxwell 解释，残差相关的有限位移提供设计判据，锁定规则提供明确的表示收益。独立盲测、普遍重建优势、加速、严格输出证书和历史优先性不属于已成立的 claim。

投稿 readiness 是另一判断，需结合最近邻原文对应、独立审阅、完整成本/候选覆盖封口和最终排版；主张状态不能由 PDF 编译成功或一个 PASS 自动升级。
