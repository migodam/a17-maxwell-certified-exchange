# 效率实验的来源与复核方法

## 范围

本目录是原 verified exchange 的独立效率实验。原 EXCHANGE_R1、CLOSURE_R1、论文、阈值和公开提交不由这些脚本改写。旧数据只作为已付费输入及历史对照；新的 policy、缓存、轨迹和失败各有独立版本。

物理核心仍为3D vector-Maxwell DDA，保留真实极化率导数、实材料 pullback、持久化物理正交材料 gauge、六照明、FP64/complex128。完整非线性比较使用相同 k=8、初始化、prior、LM、材料约束与 Armijo，最多18次更新。三次重复用于计时，不增加独立对象数。

本文中的固定状态（frozen state）指保持材料与线性化算子不变；参考降阶空间（anchor）只用于提案排序；finalist是进入完整定向核验的候选。RHS指每个照明对应的线性系统右端，多右端求解（block solve）共享同一分解。子过程计时（child wall）与外部全过程占用（occupation）范围不同。fallback指原路线不能采用时的退回处理；receipt是记录来源与哈希的运行凭据。材料度量（material gauge）规定物理材料向量的范数，不能随参数缩放更改。

## 版本与合同

| 内容 | 入口/合同 | 用途 |
|---|---|---|
| 原策略与频率消融 | `code/efficient_exchange.py`、`configs/FROZEN_V1_LOCK.json` | 原 top2/cap3 与四个明确命名的频率 policy |
| 物理 profile | `code/profile_physics.py`、`configs/PROFILE_V1_LOCK.json` | 可恢复观察包装器，exclusive 计时；不代替部署 benchmark |
| 几何 bank 缓存 | `code/geometry_cache_v2.py` | 缓存不随材料改变的 V/meta；不缓存 L、B、Jd |
| 缓存及 block 对照 | `code/verify_cache_block_v2.py` | 实际 MISS/HIT、逐数组核对、同 RHS 数 block/column 计时 |
| 主要重建矩阵 | `code/run_nonlinear_efficient.py`、`configs/NONLINEAR_MAIN_V1_LOCK.json` | 两对象、四路线、三次顺序轮换计时，共24个独立冷启动子进程 |
| previous-space pilot | `code/run_nonlinear_reuse.py`、`configs/NONLINEAR_REUSE_V1_LOCK.json` | 两对象、两种复用规则，每种一次；不混作三次 matched timing |
| top1/cap3 secondary pilot | `code/run_nonlinear_top1cap3_v1.py`、`configs/NONLINEAR_TOP1CAP3_V1_LOCK.json` | 原先声明的一名候选、三次交换；独立 CONDITIONAL 授权，不冒称 Gate A GO |

缓存V1和分析V1/V2保留在目录中，不能用新文件反推旧版本正确。缓存V1的数值相同但没有实际hit，未获准部署；V2须同时通过来源核验及实际命中核验。分析V3识别三种 nonlinear driver 的明确版本和源码 SHA，按配置/来源/方法/重复分层。

previous-space试验只复用物理电流空间，新材料状态仍重新获取L、B及其作用，不复用旧Jx/Jd。top1/cap3策略预先列出，但其补充重建试验在观察到top1/cap2问题后才准入；保留CONDITIONAL身份，不当作独立holdout。

## 实际执行版本的绑定

每个任务有独立 config 哈希、源码哈希、输入 manifest、baseline lock、实际导入模块路径/哈希、数值库版本和设备。runtime adapter 读取严格锁定的59文件依赖 clone；其中17个项目文件和42个 vendor 文件不能由当前工作树随意替换。clone 用于运行，原目录用于核算历史费用，二者职责不同。

watcher 连续记录启动、结束、占用时长和显存。启动前同时检查源码与输入、配对账本、现有进程、执行锁和剩余累计预算。连接丢失或未知终态不得自动重试。所有文件保持原版本，终态快照以 SHA 绑定；仅有退出码或一份哈希不作为科学验收。

私有SSH连接保存在已有 transport 配置中，不是数值方法的一部分。这里不发布凭据、host-bearing argv 或 private stderr，也不对可执行源码做全文正则脱敏。

## 本地复核

在项目根目录使用现有 `Gaussian/.venv_nn/bin/python`。这些步骤只读既有原始结果，不建立新的物理状态，也不调用GPU：

```text
Gaussian/.venv_nn/bin/python Gaussian/A17/EFFICIENCY_R1/code/analyze_efficiency_v3.py \
  --snapshots Gaussian/A17/EFFICIENCY_R1/private_audit/final_evidence_v2b/private_audit \
  --execution-receipts Gaussian/A17/EFFICIENCY_R1/private_audit/final_evidence_v2b/private_audit/research \
  --out Gaussian/A17/EFFICIENCY_R1/analysis/RECHECK_DERIVED_FRESH

Gaussian/.venv_nn/bin/python Gaussian/A17/EFFICIENCY_R1/code/audit_nonlinear_evidence_v2.py \
  --data Gaussian/A17/EFFICIENCY_R1/analysis/RECHECK_DERIVED_FRESH \
  --raw-root Gaussian/A17/EFFICIENCY_R1/private_audit/final_evidence_v2b/private_audit \
  --out Gaussian/A17/EFFICIENCY_R1/analysis/RECHECK_RAW_AUDIT_FRESH

Gaussian/.venv_nn/bin/python Gaussian/A17/EFFICIENCY_R1/code/assemble_efficiency_cost_v2.py \
  --snapshot Gaussian/A17/EFFICIENCY_R1/private_audit/final_evidence_v2b \
  --out Gaussian/A17/EFFICIENCY_R1/analysis/RECHECK_COST_FRESH

Gaussian/.venv_nn/bin/python Gaussian/A17/EFFICIENCY_R1/code/plot_efficiency_final.py \
  --data Gaussian/A17/EFFICIENCY_R1/analysis/derived_v3 \
  --out Gaussian/A17/EFFICIENCY_R1/analysis/RECHECK_FIGURES_FRESH

Gaussian/.venv_nn/bin/python Gaussian/A17/EFFICIENCY_R1/code/plot_owner_efficiency_pareto_v1.py \
  --audit Gaussian/A17/EFFICIENCY_R1/analysis/owner_nonlinear_v2/AUDIT.json \
  --analysis Gaussian/A17/EFFICIENCY_R1/analysis/derived_v3 \
  --out Gaussian/A17/EFFICIENCY_R1/analysis/RECHECK_OWNER_FIGURES_FRESH
```

若本机快照目录名不同，使用最终证据清单绑定的 raw-root。缺少数组时审计必须显示 NOT_MEASURED/INCOMPLETE，不能仅凭 JSON 标签宣布材料约束或缓存等价通过。公开/可移交的数据表可独立重画图；没有私有依赖和物理输入的环境不能声称完成全GPU重放。

上述analyzer与数组auditor必须指向同一份最终快照的private_audit子目录；cost assembler指向包含FINAL_EVIDENCE_SUMMARY_V2.json的外层目录。费用汇总入口为analysis/final_cost_v2/BUDGET_RECEIPT.json：旧carry只加一次，成功preflight与monitor已经包含于occupation；没有启动任务的拒绝guard另列。缺失/未配对记录保持OPEN，不补零。V2仅在同一watcher源码与显式不存在的inventory、配对attempt及预算一致时识别无rejected-preflight；原V1的OPEN结果保留。cost assembler拒绝覆盖已有输出；再次复核时应指定不存在的--out目录。大写RECHECK目录是建议的新复核目的地，不代表已有结果；这些命令不发起GPU实验。

最终收集器先要求所有已开始的串行队列具有可审计 clean terminal，再检查远端无活动/未知数值进程、执行锁或未配对attempt。收集前后源码、配置、账本和文件库存须相同。大量 `public_workspace.npz` 仅留不可变远端哈希；其他核验、final、checkpoint数组单独同步。因此这是 evidence/reanalysis package，**不是完全自包含的GPU replay包**。

## 不同的成本范围

1. frozen policy wall 是已获取公共映射后的边际费用；不能当作 cold reconstruction。
2. nonlinear child wall 包括计时范围内物理 setup、各阶段、数组存档及 final metrics；它不包括开始计时前的进程启动/导入/守卫或结束后的最终发布。
3. external occupation 另外包括启动、监控和退出，用于累计12小时预算。不能与 child wall 相加。
4. profile 的 Jx/Jd 和 bank 子区间嵌套于总体。只在正确层级累加 exclusive 区间；不把全部计时表扁平求和。
5. 旧输入生成/teacher 是历史已付费用；新任务不将它们称为免费从零获取。失败、候选拒绝、fallback 和诊断物理作用均保留。

## 门槛

`configs/POLICY_PRELOCK_V1.json` 在新结果前固定：每对象冻结总收益保留≥90%且平均Jd≤2为 Gate A GO；nonlinear material/held-out误差不得超过原策略的1.01倍再加5×10^-9；实际接受必须满足原约束和Armijo。runtime目标是 exchange/receiver≤1.3，强目标1.15。secondary CONDITIONAL pilot、不完整覆盖和单次实测均单列，不能自动升级主门槛。

最终科学判定由主线程对原始账本、数组、曲线与适用条件共同审查；脚本只产生数据与问题清单，不给论文成功背书。

## 终态和实际秩

最终34条重建轨迹及90个新增冻结endpoint全部闭合。analyzer receipt仍为PARTIAL，是泛用三次重复期待与P6单次/历史单次不一致的标记，不是未知GPU进程。owner coverage明确保留这一区别，不改写脚本状态。

k=8是六个illumination共享空间的8个**复电流列（complex current columns）**。实化后的电流坐标可有16个实自由度，但材料coordinates始终是实数物理坐标；不能把16当额外电流列数。Gaussian与voxel材料维数不同，使用各自已锁定物理度量；跨表示数据不拼成同一参数矩阵。

## 可移交包的重分析

本地原始数组审核与公开表格重画是两个权限层次。`analysis/public_reanalysis_v1/`只把来源绝对路径结构化映射成声明的相对证据命名空间，数值字段、源码与NPZ payload不作全文清洗。输入/输出SHA及变更字段有独立凭据。相对证据名不表示1.3GB原始数组或10.8GBworkspace已随包分发。

解压证据包后，在已具备NumPy/Matplotlib等依赖的环境，从包根目录可重画：

```text
python code/plot_efficiency_final.py --data analysis/public_reanalysis_v1/derived_v3 --policy configs/POLICY_PRELOCK_V1.json --out RECHECK_PUBLIC_FIGURES
python code/plot_owner_efficiency_pareto_v1.py --audit analysis/public_reanalysis_v1/owner_nonlinear_v2/AUDIT.json --analysis analysis/public_reanalysis_v1/derived_v3 --out RECHECK_PUBLIC_OWNER_FIGURES
```

两条命令只读已审计表，不形成Maxwell作用。compact RAW_AUDIT_SUMMARY供审阅，owner绘图必须使用完整schema的投影AUDIT.json。全source/data审计和GPU重放仍需要私有完整快照与42个external vendor模块；不能在只有证据包的环境运行上述私有复核命令后声称全重放成功。
