# A17 可复现性、执行版本与证据边界

这份交付是 **evidence/reanalysis package（证据与再分析包）**。本地保存了真实物理执行所需的内核与输入；公开包不宣称 self-contained GPU replay（自足 GPU 重放）。42 个 hash-identified external/vendor modules 的公开分发权尚未关闭，因此公开包提供其版本身份、实际导入证据和要求，不以缺少这些模块的公共机器重跑充当物理验收。

## 1. 什么已被实际核验

冻结方法源码、在线容差、配置、材料输入、receiver/candidate/anchor 数组和每次执行的 runtime receipt 共同绑定执行版本。哈希核对关注“这份代码被哪个作业实际加载”，而非仅证明某目录里存在相同文件。

- 保存证据审计完成 1,749 项来源与数组绑定，未发现失败；其中 42 项外部模块字节、133 项材料输入、75 项回放接受步逐数组比较、36 项终态风险和 1,080 项实际导入模块绑定。详见 `research/delegated/a17_closure_repro_final/HASH_CHECKS.json`、`SAVED_REPLAY_AUDIT.json`。
- 156 个 Python 文件 AST 解析通过。唯一类似 private 命名的 `private_interface_checks` 是正常测试变量；不存在 `PRIVATE_SSH_ROLE` 等可执行数值表达式替换。
- 105 个现有 CPU 测试通过，覆盖在线容差、正常/异常 collector、固定协议、作业预算与停止逻辑和绘图接口。它们属于合成/保存证据测试，不是新增 Maxwell 场景。
- 六个随机复小矩阵试验核验 Schur swap、端点步差、双侧支撑、真实位移的二次收益、复数 directional Q、相位及排列不变性。最大记录误差量级为 $4.24\times10^{-14}$。这些代数试验没有被升级为独立物理证据。
- 实际 CUDA 小 Maxwell smoke v3 保存了 35 项检查、真实导入路径和源哈希；closure 三个关键实现及该执行版本哈希一致。CPU smoke v1 的测试脚本身份已过时，不能反向代表当前源码；CPU v2 保存的三项主实现哈希相符，仍保留各次不同的 runtime 版本。
- 原 36 单元物理重放验证在线阈值替换：75 个接受动作与每轮材料步/输出、最终风险保持一致。旧评估 floor 没有回流进入部署接受。

新增加的 collector、图和公共准备工具的纯离线测试单列保留，不能与旧测试数混在一起宣称全部内核或所有物理场景均已独立认证。

## 2. 必须区分的源码层次

|层次|作用|保存与核对方法|
|---|---|---|
|历史 EXCHANGE_R1|原四对象、36 单元与机制见证|不改原文件及原公开提交；回放逐源与逐动作绑定|
|closure 执行源|在线阈值、冻结验证、noise、nonlinear、offline coverage|作业 start/end receipt、实际 imported paths/hashes、配置与输入哈希|
|当前写作/分析源|表格、图、报告和 LaTeX|分析输入 authority、源哈希及输出哈希；不改变物理执行|
|public reanalysis port|在公共相对路径布局读取已保存证据|独立命名和源身份；路径/import 可移植修改明确列出，不冒称原执行源逐字相同|
|外部物理模块|DDA、材料 gauge、切向/伴随作用|42 项模块身份及实际加载子集；未授权再分发的模块不复制公开|

远端执行基线是 13 个基线 code 文件，本地另有 `audit_first_state.py`，共 14 个。严格重放应使用远端已绑定的 13 文件 epoch，不能删除版本检查来掩盖额外文件。首次 pool-hash mismatch、过时 smoke 版本及其费用均保留；修正后使用独立版本和记录，不覆盖失败。

## 3. 已保存数据能支持的重现

不访问 GPU 即可重算：原回放动作/数组的一致性；额外对象和噪声的绝对 GN gap、对象内求和比例及 bootstrap；真实秩、solver residual 和失败分母；六条完成的 nonlinear 轨迹与一条 17-update 前缀的可行性、Armijo、字段和计数；成本与候选覆盖的保存证据统计；所有论文数据图。

重新执行完整物理方法则还需要相符的外部内核、锁定 FP64 backend、DDA 约定、原始 material gauge、冻结输入及原运行配置。公开 `START_HERE` 区分这两条路径。缺内核时应报告 `BLOCKED_MISSING_KERNEL`，不得用随机数组替代 Maxwell 测试。

## 4. 快照与保存证据

所有数值阶段都在作业终态、无未配对 attempt 和无共享执行锁后提取不可变快照。scoped composite snapshot 保留具体包含哪些阶段，并绑定父快照中已经关闭的 replay/primary/noise/inputs。后续范围更大的快照不能被当成旧作业当时加载新代码的证明。

|已关闭快照|文件数|ZIP SHA256|
|---|---:|---|
|primary_complete_v1|2006|`60e2ff89601be6a1e0ab5a8de13a79ec35f96199a2cdcd5b26ed90b635206641`|
|noise_complete_v1|3113|`d613ac38eca5b6244bbc8c8bb8e1f5f823ee489588310bb4963f74fc61d6670e`|
|nonlinear_phase_closed_v1|1444|`7f05f3bf5b588bb9c3a67daf97997c64deea26e495b25ad6a99c5185919c6ad3`|

传输超时与数值失败分开记录。noise 归档第一次传输超时后，确认远端已经完成同一 ZIP，再以相同哈希取得文件；没有重跑物理作业。独立 reanalysis 会核验 ZIP、member、receipt 和阶段范围，不能仅凭状态字段 `COMPLETED` 或退出码零接受数据。

## 5. 公开导出规则

可执行研究源禁止全文正则清洗。数学表达式、矩阵乘法、共轭和变量名必须逐字保存；隐私配置不进入它们。需要公开适配的分析工具按独立 port 保存，逐项解释 import/相对路径变更，并运行合成 fixture 与 AST 对照。

元数据采取逐字段审查，而不是全文替换。已审查的 12,064 个路径字段和 24 个路径键仅指向用户明确给出的研究目录或已知解释器目录，没有凭据、SSH 私密配置或未知路径。这些具体文件字节经独立批准后原样保留，以免改写 runtime receipt 的内部哈希关系。批准只绑定已审查的文件哈希；新增终态快照必须另行检查。原始 SSH/CIM 进程命令日志、凭据、第三方全文 PDF 和无明确分发权的模块不进入公开包。

二进制证据以 SHA256 去重并保留每个原路径 alias。去重只减少字节重复，不合并实验样本、覆盖范围或费用。完整输入与逐动作数组、失败前缀仍可按 manifest 恢复。公共包明确区分原样执行源、经批准保留的元数据、排除的传输/凭据材料和独立 reanalysis port，而非笼统写“所有源文件 unchanged”。

## 6. 可复现性不能替代的科学条件

小矩阵测试、解析成功和字节一致不证明历史优先性、TAP 接收或任意材料/采集条件下的泛化。全参考最优步仅用于离线评估；controller 的 Jx/Jd 完整作用是收费的在线信息。归档保留一个 primary 状态的 receiver residual failure、nonlinear 的第四对未完成，以及曾被研究过的额外对象不等于完全新盲测的限制。

部署的 numerical significance tolerance 没有给出合法的 Maxwell output-error radius，因此不能写成 deterministic acceptance/stopping certificate。相同 k 不等于相同计算成本；GN-gap 改善和 nonlinear 材料误差是两种 endpoint。

## 7. 真实公开入口的保存证据复算

新 portable saved-evidence port 在独立本地布局读取原数组，不导入外部 Maxwell 内核。三完整对照的 36 项终点指标和 18 项比值均与 owner 统计精确一致，125 条 objective 曲线逐行一致，原 11 个数组分析前后 SHA 不变。严格全-case/full-metric模式因第四对象失败前缀没有 field/held 而失败；对应三项量保持 null。支持保存量的模式成功，但不冒称补齐第四组或重新验证物理。

公开费用 CSV 复算精确得到 19,962.060 秒，保留 4 次非零 attempt 与 2 次被拒 preflight，未加入 nested stages。私有 ledger 的命令/进程原件仍未公开，故这里复核的是明确审核后的费用派生，不是独立 monitor re-audit。

部署源码中的 16 个 Python 文件、11 个测试文件已重新解析；这是该执行层的语法检查，不是整个公开包的文件数量。原 CPU v2/真实 CUDA v3 smoke、exchange/Q identity、物理 replay 和固定种子记录各自保留实际 runtime/source epoch，不能由当前解析成功替代。公共 port 和执行源身份在 manifest 中分开，禁止全文正则脱敏数学表达式。

## 8. 候选覆盖的独立保存数组复核

来源collector v5实际核验32状态/96单元。独立保存数据检查进一步核对285完整checkpoint、377720声明动作、370521可行score与270个teacher接受动作。初次capture、top-one、12pool和索引重算精确一致；所存选中路径的directional Q最大差7.81e-18，96个actual-minus-teacher gap差最大9.76e-19。15个有saved dense J的状态支持90端点绝对gap复算，最大差3.25e-19。

这个检查没有重新调用Maxwell。全候选的材料方向D和输出Z未全部保存，因此没有从物理公式逐一重新生成全部370521个标签；17个无dense J状态的绝对gap未独立复算。由保存Js相减恢复的Z低位不能等同于原输出字节hash，270方向hash精确一致但不宣称输出hash也一致。完整细目列于coverage_array_review_v1。

最终统一账本314行保留失败、缺失与未运行，不将314行称为独立对象。数值campaign61个attempt全部配对闭合；公开费用由安全CSV派生复算，私有monitor/命令不公开，public CSV复算不等同于私有monitor再审计。
