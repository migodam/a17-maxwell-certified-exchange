# A17 失败、负结果与适用条件

正面结果与以下边界共同构成论文证据。原始失败不回写成 abstention，不用最后安全状态伪造完成终点，不通过放宽阈值补齐覆盖。

## 当前实际未完成结果

|阶段|计划分母|已审计/已完成|失败和未运行|处理|
|---|---:|---:|---|---|
|原36单元 threshold replay|36|36；75接受保持|无结果变化|原数据保留；新在线规则独立版本|
|additional-object primary|11对象、99单元|96；10对象完整、1对象6单元|非对称voxel late k=4 receiver endpoint normal residual FAIL；k=8/16 NOT_RUN|不换对象、不降低阈值；partial对象统计单列|
|有限噪声|4对象、72单元|72；71improved、1no-op|shell Gaussian 3% realization1/k16 no-op，不是失败|所有被拒finalist和费用保留|
|约束非线性|4对象、8轨迹|6完整终点、3完整对照|非对称voxel exchange iteration17 FAIL，17更新前缀保留；receiver NOT_RUN|队列按合同停；不retry、补旧baseline或冒充四对|
|离线机会/有界路径|33状态、99单元|32状态、96单元完整|primary前置失败对应3单元固定NOT_RUN_PRIMARY_INCOMPLETE|只用已有96单元，局部teacher不称全局oracle|

## 被保留的科学负结果

- 等材料秩的 Krylov repair 在72/72 smooth、36/36 sharp 比较中均优于 paired repair；后者虽优于对应 direct enlargement，却没有 solver 优越性。
- 原 paid paired warm start 为 same-M0 PCG 的1.416倍wall；full-wave RHS由66增至342。闭包存在不保证取得该空间便宜。
- A16宽域from-scratch选择的对象风险比中位1.0947，只6/15对象改善；A17限制到receiver附近的条件替换，没有推翻这一失败。
- 三对完整非线性中，近接触对象的复误差改善仅0.22%，实部从0.6810升至0.6835。更低data residual并不表示每个材料分量都更好。
- 三对all-in child wall均更长，比分别2.68、1.74、1.46。没有成像加速结果；一次计时不形成独立runtime重复区间。
- 四个历史sharp全模型误差为0.755、0.750、0.560、0.906；粗高对比球的Mie偏差21.40%仍保留，不能由新交换覆盖。
- 五个历史归一化closure检查失败仍按原验收保留。良态Schur和极小绝对distance支持相消/小分母解释，未证明underflow，更没有事后更改PASS标签。

## 实现和审计事件

1. **原在线阈值的reference-floor依赖。** 已确认并修掉部署定义，历史源码与结果不改。完整物理replay证明75接受和终态数值不变。新FP64 tolerance只是数值显著性，不是物理误差证书。
2. **串行父作业末尾失败导致collector误拒完整前缀。** v2 collector只接纳有完整来源/数组/已配对账本的成功child，且要求唯一失败在末尾。保留旧分析和父作业全费用；未改数值结果。
3. **raw executed_step_norm 的拒绝路径字段缺陷。** Armijo全拒绝时记录下一未检验alpha的提案长度，实际材料不变。派生审计按保存状态计算实际接受更新，拒绝应为0。当前125保存更新均接受，缺陷未触发，冻结源码未改。
4. **local launcher绑定/传输中断。** 不兼容prepared解释器的一次准备尝试在启动数值之前停止；随后使用同一prepared解释器一次启动。曾有noise ZIP传输等待超时，原archive进程和完整ZIP经终态及哈希核对后取回；没有由传输问题重跑数值。
5. **历史smoke与当前源版本不同。** CPU v1只作历史记录；CPU v2及CUDA v3与声明版本分别绑定。锁定的13文件baseline与当前14文件inventory不同，不能无说明把当前工作树当历史执行源。
6. **最终saved-only collector的格式兼容。** v1–v4的来源收集先后对Windows路径分隔、legacy archive manifest位置和vendor路径映射作了过严格式假设，全部失败输出保留。v5仅规范同一绝对路径的分隔符并拒绝重复/冲突，保持root、集合与SHA约束，实际32状态通过。旧失败不是新数值重试，也没有更改研究源码或科学阈值。
7. **公开复算的严格模式缺量。** 第四条失败非线性前缀没有完整终态field/held记录，严格全指标模式明确失败；支持模式保持null。三对完整终点36项指标和18项比值精确复现。缺量不填零、不用未完成轨迹冒充成对终点。

## 证据可支持到哪里

固定k控制的是最终共享复电流维数，不控制辅助anchor/dictionary或筛选费用。完整forward residual与真实材料注入共同，变化的是tangent-current policy。主终点是冻结完整GN quadratic gap；约束、line search及重新线性化后的最终truth另验收。

额外对象曾进入A16研究。没有找到其用于所查A17 teacher/NN或对象特定调参，但历史间接影响不能排除。故严格新blind对象数未确认；对象bootstrap描述固定资产集合，不能转成无分布条件的泛化保证。

短池外动作无合法收益上界。没有好finalist只说明这次短池中的检查未接受，不能说明全字典没有有用方向。离线dictionary teacher与local path用于记录coverage、scoring和path interaction，不进入线上决策，不构成global optimum。

受控dark-current和信息/收益反例给出可能机制；没有把每次已接受收益因果分配给暗电流、强feedback或某个pullback channel。Pair-positive见证不是同秩swap trap，不代表通用非次模概率结论。线性/二次mandatory-choice的29/36与1/36不是deployment failure rate。

独立Mie只关闭指定moderate sphere及uniform-material derivative；非径向对象和全部voxel tangent仍以DDA离散证据为主。42外部vendor模块的hash可核对，不等于已获得公开重分发权或自足GPU包。
