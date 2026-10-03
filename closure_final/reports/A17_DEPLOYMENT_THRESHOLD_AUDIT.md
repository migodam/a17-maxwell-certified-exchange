# 接受阈值的部署来源审计

历史实现存在一项部署依赖：排序不需要完整最优步，但接受阈值读取了由完整参考解计算的 numerical floor。该依赖现在通过独立版本的在线数值阈值移除；是否改变原 36 个单元的决定，另由实际 replay 验收，不能由预计替代。

## 历史链与两种实际执行版本

`EXCHANGE_R1/code/maxwell_state.py:54–66` 用保存的完整步、其完整方向响应与伴随正规残差计算 `EvaluatorContext.floor`。`run_exchange.py:99–102,151,161` 将 `10*floor` 放入接受容差。物理 matched timing adapter 读取保存的 floor，也保留了它的完整参考来源。

36 个单元并非全用同一阈值源码：近接触 Gaussian 初始状态的三个预算加载了 SHA256 `8fa298a5f08c960e5e602916c63d786dd7d735f5c222284ffcf317f552ac3d43`，其每轮公共能量尺度为残差、材料二次项及线性 prior 项；其余 33 个单元加载 `8305891509db380224bcdc1743cb3e69c0d84cd706629afdafcf40645400448d`，使用逐候选 base/child objective 尺度。精确历史文本与 ZIP 绑定在独立审计目录中保留。两种实现都读取完整参考 floor。

历史记录的 floor 为约 $8.87\times10^{-20}$ 至 $3.81\times10^{-15}$；move 容差为约 $8.87\times10^{-19}$ 至 $3.81\times10^{-14}$。这些是 FP64 能量尺度，不是最小可表示数，也不能作为下溢证据。

## 五种量的职责

|类别|来源与用途|是否允许部署接受读取|
|---|---|---|
|A：evaluation-only reference floor|完整参考解的残差与目标量级；用于离线风险分辨率和分母标记|否|
|B：online acceptance tolerance|下述当前残差、方向输出、材料步及 prior 的 FP64 数值尺度|是|
|C：deterministic output-error radius|需要合法的物理输出误差上界；当前没有实例化|当前不存在|
|D：linear/current solve tolerance|既有电流块稳定性和端点正规残差验收；实际代码及最终冻结配置为 $10^{-10}$；旧元数据曾列较宽的 $10^{-8}$ 上限|独立质量检查，不能冒充 C|
|E：floating-point significance tolerance|操作长度、FP64 epsilon 与在线能量尺度|是，但仅是数值分辨规则|

原旧规则及其结果不修改。离线 evaluator 可以继续计算 A，但在线 controller 没有 evaluator/floor/reference/truth 参数。

## 冻结的新在线数值规则

在共享实材料坐标下，令 $u=r+J_Fx$、$z=J_Fd$，使用完整方向收益

$$Q=-u^Tz-(\ell+\lambda x)^Td-\tfrac12(\|z\|^2+\lambda\|d\|^2).$$

本实现保持原标量正则曲率 $\Lambda=\lambda I$，不修改材料度量。为避免仅以相消后的 $|Q|$ 衡量有效数字，定义

$$\begin{aligned}
S_Q={}&\sum_i|u_i||z_i|+\sum_j|\ell_j+\lambda x_j||d_j|\\
&+\tfrac12\|z\|^2+\tfrac12\lambda\|d\|^2
+\|r\|^2+\|u\|^2+|\ell^Tx|+\lambda\|x\|^2.
\end{aligned}$$

所有量均在线可得。最后四项保留当前基点的量级，使极小步差不会被赋予虚假的零舍入尺度。设 $n=m+3d+16$，其中 $m$ 是实化测量维数，$d$ 是实材料维数；$\epsilon$ 是 FP64 machine epsilon，$\gamma_n=n\epsilon/(1-n\epsilon)$。固定

$$\tau_{\rm online}=32\,\mathrm{tiny}_{64}+32\gamma_n S_Q.$$

操作长度取覆盖点积与合并项的保守计数，系数 32 在 holdout 结果产生前由主线程固定，未根据收益标签调节。该规则处理数值有效数字；它没有传播完整 Maxwell 求解误差，因而**不是 deterministic certificate**。端点/current 质量检查仍使用原来的冻结规则。

新接口 `code/online_tolerance.py` 只接收 $r,u,z,x,d,\ell,\lambda$。接受为严格 $Q>\tau_{\rm online}$；至多两个 finalist 中选取满足条件者的最高完整收益。不存在合格 finalist 时保持 no-op，并报 `ABSTAIN_UNRESOLVED`。它不证明完整字典局部最优。

## 独立开发一致性检查

仅使用原开发对象近接触 Gaussian 的三个已保存状态，重建各预算的初始 top-two 方向，并以已支付的 Gaussian Jacobian 作明确标注的离线一致性诊断。两次重复共 36 行，重复位相同；方向公式与目标差的最大绝对差为约 $1.36\times10^{-19}$，最大 error/tau 约 $3.19\times10^{-5}$。这项检查不执行新 Maxwell action，不读取新 holdout 标签，也不构成部署使用完整 Jacobian 的许可。

完整 36-cell replay 使用真实物理方向 callback，补齐旧记录没有保存的初始 $Jx$ 和 rejected finalist $d,z$；决策变化必须沿新路径运行，不能把旧的后续 neighborhood 接在改变后的路径上。

## 证据与尚未关闭事项

细粒度文件/行号/哈希、两份实际执行源码、175704 条旧 move 阈值的来源表位于 `research/delegated/a17_closure_threshold_trace/`。新的开发诊断位于本目录 `analysis/online_tolerance_dev_*`；unit tests 位于 `tests/test_online_tolerance.py`。

本审计已确认部署定义问题及其隔离方式。真实物理回放已验收：36/36 单元的75个接受动作、终态空间、每轮材料步和方向输出完全保持，最终及receiver风险差为零。详见 `A17_THRESHOLD_REPLAY.md` 和 `analysis/replay_audit_v2/`。这一关闭不把数值阈值升级成确定性输出证书。
