# 失败、未运行和收集异常均保留

## 实验中的负结果

- 解析拒绝：精确g_F(x)不是共同已付费量；保存回看仅壳体3/106可拒。部署节省0，不引入额外伴随。
- 位移压缩：同冻结状态多数满秩；不同状态J不同，不用跨状态rank宣称可复用旧Jd。
- block优化：已有真正multi-RHS；新控制略慢，不选择最快重复。
- 一次交换：壳体冻结收益仅77–78%，不满足90%门槛。
- 两次交换：冻结Gate A通过，平滑非线性质量失败，不隐藏也不作为最终建议。
- 沿用空间：always-reuse质量与时间都较差；period-3未保住平滑质量，不按事后家族结果挑策略。
- 经验triage：固定半径已存样本覆盖，但完整非线性提前拒绝0个；不将经验覆盖称certificate。
- Gate C：推荐路线1.958/1.318均未同时满足≤1.3；无普遍加速结论。

## 未运行分支

verification ROM为NOT_RUN，不是实验失败。原Jd只占几秒、位移基本满秩、半径triage没省核验；有限任务中继续构建ROM不击中主要费用。没有NN/GNN/Flow、没有新current selector、没有新的PCG/preconditioner，也没有学习模型训练。

## 工程异常与版本关系

| 事件 | 对数值结果的影响 | 处理与保留 |
|---|---|---|
| 首次profile缺旧非数值audit依赖 | 在性能实验前拒绝，1.906秒计费 | 原baseline未修补；创建59文件精确身份clone，之后独立准入 |
| cache V1实际MISS/MISS | 输出正确但没有缓存收益 | 31.250秒计费；未部署，V1原件保留，V2独立命名并真实MISS/HIT |
| continuation helper缺两份owner记录 | 在P6/补充分支启动前停下，无数值attempt | 仅部署两份精确哈希非数值记录；原helper/失败日志留存，独立readmission，无重跑数值 |
| 首次只读collector失败 | 没有改动任何实验 | 根因OPEN；timeout只是可能。same collector、fresh v2b、较长只读传输超时，最终不可变验证闭合 |
| cost assembler V1缺preflight ledger | 数据格式兼容问题导致OPEN | 原V1保留；V2要求实际watcher身份、明确不存在绑定和账目对齐，不能泛化补零 |
| analyzer generic receipt为PARTIAL | P6与历史只做n=1，不满足泛用3rep期待 | 真实34/34授权任务与规定90endpoint完整；不改写generic输出，不把其PARTIAL误报为队列仍在跑 |

原始错误/host/argv只存在私有证据中；公开报告保留原因、不确定性、费用及哈希，而不输出敏感连接资料。

## 尚未关闭的外推

两个预指定对象的无噪声轨迹不能证明泛化、新噪声稳健性、任意秩最佳策略或连续Maxwell精度。three-repeat wall range是机器计时离散程度。推荐路线是主试验质量失败后准入的预声明补充分支；没有宣称独立holdout。

生产implicit同步计数、完全分解的传输/I/O/准备时间没有完整量化；未归属区间明确保留。完整非线性每个state的GN最优步未形成，故不能给逐state GN gap。final truth更好不等于已证明每步更接近完整GN。

旧A17所有正负结果、constraints/line search边界、source/external依赖与原公开提交均保留。这里不把new policy覆盖为原verified exchange。
