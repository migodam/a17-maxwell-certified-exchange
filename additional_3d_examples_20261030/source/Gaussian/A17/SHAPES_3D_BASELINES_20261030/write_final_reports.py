"""Present existing final records; no physics, new metric solves or tests."""
from pathlib import Path
import csv
import json
import statistics

ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'analysis/FINAL_SUMMARY.json').read_text())
objects=s['objects'];cost=s['cost']
cases=['digit3','letter_e','torus','double_torus','cross3d','offset_shell']
names=dict(zip(cases,['立体数字3','字母E','单环','双环','三轴交叉杆','偏心核壳']))
policies=['full_gn','receiver','projected_internal','fourier','random','exchange']
labels=['完整GN提案*','Receiver / SOM-like','内部传播proxy','Fourier','随机空间','A17交换']

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in rows])+'\n'

def pct(x):return f'{100*x:.2f}%'
rows=[]
for case in cases:
    o=objects[case];best=min(o[p]['complex_nrmse'] for p in policies)
    values=[pct(o[p]['complex_nrmse']) for p in policies]
    values=[f'**{v}**' if o[p]['complex_nrmse']==best else v for p,v in zip(policies,values)]
    rows.append([names[case],*values])
primary=table(['目标',*labels],rows)
ratio=[objects[c]['A17_ratios']['receiver'] for c in cases]
wins=sum(v<1 for v in ratio);median=statistics.median(ratio)
ratio_table=table(['目标','A17 / receiver','A17材料误差相对下降（负值为恶化）'],
                  [[names[c],f'{r:.4f}',f'{100*(1-r):+.2f}%'] for c,r in zip(cases,ratio)])
fallback=table(['目标',*labels],[[names[c],*[f"{objects[c][p]['gradient_fallback_count']}/18" for p in policies]] for c in cases])
components=table(['目标','A17实部NRMSE','A17虚部NRMSE','A17支撑IoU','A17测量残差','A17留出误差'],
                 [[names[c],pct(objects[c]['exchange']['real_nrmse']),pct(objects[c]['exchange']['imag_nrmse']),f"{objects[c]['exchange']['support_iou']:.4f}",pct(objects[c]['exchange']['measured_relative_residual']),pct(objects[c]['exchange']['heldout_relative'])] for c in cases])
charges={}
for line in (ROOT/'COST_LEDGER.jsonl').read_text().splitlines():
    e=json.loads(line);key=tuple(e.get('stage','').split('/')[:2])
    v=charges.setdefault(key,dict(state=0,rhs=0));v[e['kind']]+=e['count']
control_rows=[]
control_csv=[]
for c in ('digit3','torus'):
    f=s['gradient_control'][c];m=f['metrics'];ch=charges[(c,'full_gradient')]
    control_rows.append([names[c],pct(objects[c]['receiver']['complex_nrmse']),pct(objects[c]['exchange']['complex_nrmse']),pct(m['complex_nrmse']),ch['state'],ch['rhs']])
    control_csv.append(dict(case=c,policy='full_gradient',status=f['status'],updates=f['updates'],wall_s=f['wall_s'],accounted_full_factorization_attempts=ch['state'],accounted_complex_full_rhs=ch['rhs'],**m))
with (ROOT/'analysis/FULL_GRADIENT_CONTROL.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(control_csv[0]));w.writeheader();w.writerows(control_csv)
controls=table(['目标','Receiver NRMSE','A17 NRMSE','普通完整梯度 NRMSE','梯度控制分解准入','梯度控制RHS'],control_rows)
wall=cost['wall_s'];hours=int(wall//3600);minutes=int(wall%3600//60)

report=[f'''# 六类三维目标的最终材料重建：A17与五种对照

**结论：这组实验不支持A17具有普遍的最终成像优势。** 在固定的六个目标上，A17的复材料NRMSE有{wins}/6低于receiver；对象比值A17/receiver的中位数为{median:.4f}，即中位误差约增加{100*(median-1):.2f}%。普通完整梯度在两个固定追加对照上都取得更低材料误差。

这并不否定原论文关于Maxwell电流改动如何影响局部材料更新的机制。它明确限制了应用结论：局部全GN步保真度提高，尚不能当作更好的最终材料图像。**我不建议用这组结果把论文改写成普遍优越的成像算法。**

36条主轨迹与2条梯度控制均完成18次接受更新，终态全部为MAX_UPDATES。它表示到达迭代上限，不表示已收敛。累计受控计算{wall:.3f}秒（{hours}小时{minutes}分钟），包括适配核验、合成、失败及接续；没有追加训练或新目标。

## 1. 先看最终误差

主要指标为最后接受材料的复NRMSE，越低越好。粗体仅表示该行六种主方法中最低值；不是统计显著性或收敛认证。

''',primary,r'''
*完整GN列准确含义是“完整GN提案＋共同约束投影／梯度保护／线搜索”，不是一个已充分收敛的最佳GN求解器。五种电流ROM的终点秩均按k=16构造；完整GN不受该秩约束。SOM-like与内部传播proxy也不是原生SOM/TSOM完整优化器。

![所有保存终态的三维外形](figures/FINAL_SUPPORT_6x7.png)

第一列标为SYNTHETIC TRUTH；后六列才是各方法的重建。图中只显示Reχ≥0.5的体素，它会隐藏弱背景误差。完整对照还应看[原始实部切片](figures/RAW_REAL_XY_central.png)和[原始虚部切片](figures/RAW_IMAG_XY_central.png)。这些图没有平滑，也没有用真值修补重建。

## 2. 这次重建的是什么

六个对象是立体数字3、字母E、单环、双环、三轴交叉杆，以及带偏心内核的空腔壳体。前五种主体χ=1+0.08i；壳与核分别为0.7+0.06i和1.4+0.12i，背景为0。形状在合成数据前固定，算法不知道支撑、边界和孔洞。

使用原三维矢量Maxwell DDA。十二次入射（±三轴、各两横向偏振）共享同一复材料；64个双偏振接收位置提供观测，另80个旋转接收位置只做离线评价。每对象一个固定的1%复噪声实现。所有方法从χ=0.1+0.01i的均匀初值开始，没有N3或其他学习先验。

12³网格含1728个复材料未知量，即3456个实未知量。电流有三个矢量分量；k=16只限制切向电流表示，**不是只重建16个材料参数**。完整forward和共同梯度仍使用完整物理。

各方法共享prior定义、LM schedule、被动性约束、可行投影与Armijo线搜索，只改变电流/GN更新提案。详细方法与指标解释见[METHODS_AND_INTERPRETATION_ZH.md](METHODS_AND_INTERPRETATION_ZH.md)。

## 3. 哪些地方确实改善，哪些地方没有

''',ratio_table,r'''
数字3有清楚的receiver相对收益，但内部传播proxy、Fourier、随机空间和完整GN都得到更低的最终材料误差。字母E上，A17反而弱于全部五种主对照。单环的receiver相对收益较小；双环和偏心核壳出现恶化。三轴交叉杆是A17在六种主方法中数值最低的一例，但它与随机空间非常接近，不能把这一细小差距当成稳健优势。

**负面判断并非完全依赖GPU阶段。** 数字3和字母E的十二条主轨迹均在最早的Mac CPU阶段结束，已经呈现“A17对receiver有时有增量，但没有全面胜过更强对照”。后续运行异常不能解释掉这部分证据。

字母E还有一个值得注意的现象：完整GN测量残差约8.71%，Fourier约13.87%；材料NRMSE却分别约59.93%和43.87%。测量拟合更好与材料真值误差更小不是同一件事。

## 4. 为什么必须加入普通梯度控制

原共同外层会在投影后的提案不下降时切换到完整材料梯度，再做相同线搜索。这个保护分支有时占据了大部分更新：

''',fallback,r'''
因此，Fourier或随机空间形成的好图像，不能全部归功于相应电流基底。完整GN在交叉杆上也有17/18次梯度回退，其差结果不能代表所有约束GN算法。

我在发现这个现象后，固定追加数字3和单环的普通完整梯度控制；它使用原来就存在的保护方向，不构建电流ROM或完整Jacobian。这个对照是post-hoc归因诊断，不能把不同历史轨迹当成唯一组件的因果证明。

''',controls,r'''
![原始切片与普通梯度控制](figures/NATIVE_XY_MAIN_AND_FULL_GRADIENT_CONTROL.png)

普通梯度的最终材料误差在这两个对象上都更低。不过，它各需要86次full分解准入，不能只因RHS较少便称总成本更低；后端也不同。本组没有建立相同总wall下的优越性。

## 5. 外形接近，不等于复材料接近

''',components,r'''
所有A17终点的虚部相对误差都很大。举例，数字3的支撑IoU约0.95，但复材料误差仍约55.6%，损耗分量误差约252%。只看Reχ≥0.5的三维图，会高估材料恢复质量。

偏心核壳的真实内核仅占8个网格点。原始切片显示，外部轮廓与内部高对比小核的恢复是两个不同问题；即使Fourier的全局NRMSE最低，也不能由此宣称小核幅值被准确恢复。A17该例有明显背景伪影、位置和幅值失真。

我逐图看过全部六种目标的三维终态与原始实部切片，并检查了虚部和另外的原生切面。这里没有“几乎完美重建”的依据。

## 6. 这对A17论文意味着什么

原论文问的是：给定当前材料状态和残差，替换一个电流方向，什么时候能改善这一刻的完整材料二次模型？其入口是J=SL⁻¹B，并通过材料注入、保留电流反馈和接收返回追踪具体空间改动的后果。

本实验执行的最终更新还经过材料约束和线搜索。A17核验的端点步差是d=s_new−s_receiver；真正执行的是αΠχ(s_new)。之后还要继续改变材料状态、总场和残差。因此，冻结二次模型中的正收益不提供最终真值误差的排序保证。新结果展示了这个边界，而不是推翻已有代数恒等式。

我的判断是：**机制和局部表示设计仍是论文应该保留的主线；一般性成像优越性没有建立。** 这批图可以作为补充应用和限制说明，不宜挑出交叉杆或数字3单独作为“全面优于GN/SOM”的主证据。

## 7. 时间、运行异常与证据强度

工作先在Mac CPU运行，后按用户要求迁移到i9-14900HX和RTX4060。远端CPU数值库4线程，进程调度范围限定为已识别的P-core；完整求解和批量端点使用CUDA FP64。并非全流程所有算子均在GPU，基底构造、部分材料代数和日志仍在CPU。

一次原生access violation和一次日志命名空间异常均已保留。相关轨迹从最后接受检查点在新进程接续，未重复已接受更新，也未改选择规则。**日志异常的根因和完整数值影响范围尚未确认，不能称为“已证明仅影响日志”。** 新GPU结果应连同这一限制使用，不替换原A17冻结主统计证据。

真值与反演均为12³同网格；仅一个噪声实现；全部方法达到18次上限而非共同收敛；硬件和执行版本在中途迁移。这里是有完整记录的受控应用探索，尚非独立连续模型认证、噪声鲁棒性总体试验或公平端到端性能基准。

完整费用、失败原产物与来源说明见[COST_AND_FAILURES.md](COST_AND_FAILURES.md)，重画和源码入口见[REPRODUCTION.md](REPRODUCTION.md)。旧论文及历史记录未被覆盖。
''']
(ROOT/'RESEARCH_REPORT_ZH.md').write_text('\n'.join(report))

start=f'''# 六类三维目标：A17材料重建完整对照

**36条主轨迹＋2条普通梯度控制已结束。全部18次接受更新，均为MAX_UPDATES，未宣称收敛。**

A17对receiver：{wins}/6材料误差更低，对象比值中位数{median:.4f}。强对照显示它并非普遍更好的成像器。普通梯度在两个固定追加对象上均有更低NRMSE。

## 最终复材料NRMSE（越低越好）

{primary}

*五个电流ROM为k=16；完整GN不限k。Receiver是SOM-like电流空间对照，内部传播是TSOM-style proxy，均非原生完整算法。所有主方法都含同一完整梯度保护。粗体只标主表每行最低值。*

## 阅读与图像

- [中文研究报告：结果、强对照及论文价值判断](RESEARCH_REPORT_ZH.md)
- [三维终态全图](figures/FINAL_SUPPORT_6x7.png) · [原始Re切片](figures/RAW_REAL_XY_central.png) · [原始Im切片](figures/RAW_IMAG_XY_central.png)
- [普通梯度控制的实际切片](figures/NATIVE_XY_MAIN_AND_FULL_GRADIENT_CONTROL.png)
- [逐方法CSV](analysis/FINAL_METRICS.csv) · [控制CSV](analysis/FULL_GRADIENT_CONTROL.csv)
- [方法含义](METHODS_AND_INTERPRETATION_ZH.md) · [完整费用与故障](COST_AND_FAILURES.md) · [源码与重画](REPRODUCTION.md)
- [给网页版GPT的审阅入口](GPT_REVIEW_CONTEXT.md)

所有图来自保存的实际材料数组；SYNTHETIC TRUTH第一列是合成目标，不能当作重建。统一阈值外形会隐藏背景误差，须一起看原始切片和完整NRMSE。

累计受控计算{wall:.3f}秒，full分解准入{cost['states']}次、复full-solver RHS {cost['rhs']}列。包含失败与接续，未训练网络。2小时上限经用户明确批准扩为3小时，累计费用未重置。

本补充研究使用12³同网格和单个1%噪声实现，且经历未闭合根因的运行异常；不替换原论文的冻结主证据。完整结果与失败一起公开，原稿不变。
'''
(ROOT/'START_HERE.md').write_text(start)

cost_rows=[]
for c in cases:
    for p in policies:
        v=objects[c][p]
        cost_rows.append([names[c],p,v['accounted_full_factorization_attempts'],v['accounted_complex_full_rhs'],f"{v['wall_s']:.3f}",v['counter_scope']])
sessions=table(['phase','PID','记录状态','占用wall秒'],[[v['phase'],v['pid'],v['status'],f"{v['wall_s']:.3f}"] for v in cost['sessions']])
cost_text=f'''# 完整费用与真实运行故障

## 累计资源

- 实际受控计算wall：**{wall:.6f}秒，{hours}小时{minutes}分钟**；用户批准上限10800秒。
- full矩阵分解准入（账本states）：**{cost['states']}**，上限4096。
- 复full-solver RHS列数：**{cost['rhs']}**，上限150000。
- 主任务36条，追加控制2条，全部MAX_UPDATES且18次接受更新。没有训练、追加形状、改噪声或放宽数值门槛。

这是本独立补充研究的累计账本，不含旧论文和旧三维研究费用。导入前启动、网络传输、打包和离线绘图不属于受控求解wall；一次缺依赖的启动在建立Budget前失败，状态/RHS为0，其启动wall未单独计量。不能把此数字叫完整用户等待时间或纯GPU-active时间。

## 全部受控阶段

{sessions}

并行阶段按master进程占用计费，子轨迹wall不相加。23388的原生崩溃没有正常close；按系统事件的进程生存时间取上整再加1秒，保守计281秒。其他正常关闭阶段保留原记录。中断tail和接续重新构建当前物理状态的费用未扣除。

## 每条主轨迹的实际费用

{table(['目标','方法','full分解准入','复full RHS','子轨迹wall秒','模型counter范围'],cost_rows)}

前两列物理计数从全局原始账本的case/policy stage汇总，包含原前缀、失败尾部与接续。模型counter若标continuation_only只覆盖新进程，不能据它认定原前缀免费。中断轨迹wall含保留前缀的原记录或保守时间估计。墙钟跨Mac、i9、CUDA CPU端点/CUDA端点以及串并行阶段，**不可直接构造算法加速比**。运行中还观测到其他CPU任务；没有干预它们。

普通梯度两条各有86次full分解准入、1248个复RHS。数字3wall {s['gradient_control']['digit3']['wall_s']:.3f}秒，单环{s['gradient_control']['torus']['wall_s']:.3f}秒。它们的更低材料误差，不自动等于更低总计算成本。

## 真正故障与交接应分开

1. **首次远端导入失败**：缺threadpoolctl，尚未建立Budget，无状态/RHS；复制本机已有纯Python模块后启动，不安装新框架。原日志保留。
2. **PID23388原生崩溃**：Windows事件记录python310.dll access violation 0xc0000005；未找到Python traceback。当时是双线程任务进程。根因未确定，不能仅凭并发时序归因于CUDA或线程安全。保留全部完成轨迹，从cross3d/exchange的chi09和full_gn的chi03接续，未重复已接受更新；后续单GPU作业内仍保留批量多RHS。
3. **PID12264日志命名空间异常**：offset_shell/exchange在4次接受更新后，clean中的np.ndarray访问报告np为builtin function/method。静态阅读没有找到正常的显式覆盖路径；根因和数值影响范围未确定，不能宣称只影响日志。原final/error/partial完整存档；换新进程从chi04接续，未改公式、候选或门槛。未提交的交换轮重新计算并收费。
4. 若phase状态ERROR来自REFUSE_EXISTING_RESULT及HANDOFF_RESERVATION，是在下一条未开始任务处安全退出的硬件/进程交接；它不是该目标的数学失败。原状态保留，不把它改成PASS。

这些恢复没有增加验证测试或bench，也没有给运行异常的旧前缀补独立数值背书。新研究结果的这一限制与原A17论文证据分别处理。静态故障说明附在[LOGGING_NAMESPACE_STATIC_NOTE.md](LOGGING_NAMESPACE_STATIC_NOTE.md)。

## 硬件实际使用

Mac CPU阶段和i9 CPU阶段后，完整DDA求解切换到原CUDA complex128路径；之后压缩材料端点也切换到CUDA complex128/float64，并批量处理同轮候选。远端数值库4线程，进程affinity范围为已识别的P-core逻辑处理器。双轨并发发生原生退出后，改为一个GPU作业；没有取消作业内批量。

运行快照曾观察到GPU约69%及100%利用率，也有CPU构造/投影/日志期间的0%快照。没有测量完整GPU-active积分，不能把峰值或瞬时空闲当作整体利用率或性能结论。
'''
(ROOT/'COST_AND_FAILURES.md').write_text(cost_text)

claim='''# 本补充研究可以与不可以得出的结论

|状态|结论|依据与范围|
|---|---|---|
|numerically supported|真实三维Maxwell重建可产生可识别的数字、字母、环和交叉结构|保存的3D复材料数组；无学习初始化；同网格12³|
|numerically supported|A17相对receiver在数字3、单环、交叉杆有较低终点材料NRMSE|固定18次更新的原记录；不是所有目标|
|failed|A17普遍改善这组六目标的最终材料误差|只胜receiver 3/6，A17/receiver比值中位数1.0328；E、双环、核壳恶化|
|failed|A17在本组全面胜过其他电流空间或完整GN提案|主表仅交叉杆数值最低，且与随机接近；两个梯度控制均更准确|
|not established|同总计算成本下的成像优势或加速|分解/RHS与墙钟须分开；混合后端/硬件；控制不是匹配时间试验|
|not established|与native SOM/TSOM完整优化器的胜负|只比较SOM-like / TSOM-style电流空间＋共同GN外层|
|not established|连续Maxwell独立认证、噪声总体鲁棒性、细小内核准确恢复|同网格、一个噪声实现、有限分辨率|
|not established|运行异常已被证明仅影响日志|根因及数值影响范围未确认；fresh接续不是独立验证|
|preserved|原论文电流改动到局部材料步的机制与条件设计口径|本研究未改变旧定理和冻结统计，也没有用最终材料误差替换旧GN-gap指标|

只有新增应用记录来自这组实验；没有重新证明、重新核验或升级原论文全部理论。若要把新GPU应用图提升为正式主文证据，应先关闭已报告的运行异常，再明确共同收敛和更强优化基线；本次没有为此启动额外重算。
'''
(ROOT/'CLAIM_SCOPE.md').write_text(claim)

context=f'''# Web-GPT review entry: six 3D reconstruction examples

This is an additive application study, separate from the frozen A17 paper. Read START_HERE.md and RESEARCH_REPORT_ZH.md first, then data/analysis/FINAL_METRICS.csv and the raw real/imaginary slices.

- 6 fixed geometries × 6 policies; 36 primary trajectories, all18 accepted updates/MAX_UPDATES. Additional posthoc full-gradient controls on digit3 and torus, also18 updates.
- True 3D vector-Maxwell DDA, same truth/inverse12³, 12 shared-material illuminations,64 dual-polarized receivers,80 held-out positions,1% fixed complex noise per object. Uniform initial material; no network or truth support.
- Receiver, projected-internal, Fourier,random,A17 endpoint current rank16. FullGN is unrestricted-current GN proposal plus the same feasibility/Armijo/full-gradient safeguard; receiver/internal are SOM-like/TSOM-style proxies, not native algorithms.
- A17 beats receiver on{wins}/6 final material NRMSE values; median A17/receiver={median:.8f}. It is not generally superior. Digit3/E CPU results already show this boundary before later Windows failures.
- Full-gradient controls: digit3 {s['gradient_control']['digit3']['metrics']['complex_nrmse']:.8f}, torus {s['gradient_control']['torus']['metrics']['complex_nrmse']:.8f}; A17 {objects['digit3']['exchange']['complex_nrmse']:.8f}/{objects['torus']['exchange']['complex_nrmse']:.8f}. Grad controls each pay86 full factorizations and1248 RHS; do not infer lower total cost from RHS alone.
- All main methods can use the common full-gradient safeguard; raw fallback counts are reported. These are hybrid pipelines, not18 pure ROM/GN updates.
- Controlled wall {wall:.6f}s, full factorization admissions{cost['states']}, complex full-solver RHS{cost['rhs']}. Mixed hardware/backends; no matched end-to-end speed claim.
- One native access violation and one unexplained helper np namespace exception are preserved. Three trajectories continued from committed prefixes with unchanged rules, all cost retained. Root cause and numerical impact are NOT resolved; do not assert a logging-only issue or new GPU certification.

Please evaluate final-image evidence separately from the frozen-paper full-GN-gap mechanism. In particular: explain why Q>0 before projection does not order nonlinear material truth errors; examine strong controls and gradient fallback; avoid treating thresholded geometry as quantitative complex-material accuracy. Public source is copied without executable regex redaction, and no additional physical rerun/check was performed for publication.

Source: source/Gaussian/A17/SHAPES_3D_BASELINES_20261030/run_comparison.py, original helpers/model dependencies in the same relative layout; collected counter fields and plotting sources are included. REPRODUCTION.md explains replay scope. Complete raw evidence download is listed in START_HERE.md after publication.
'''
(ROOT/'GPT_REVIEW_CONTEXT.md').write_text(context)
resolved=json.loads((ROOT/'CONFIG.json').read_text());resolved['wall_cap_s']=10800
resolved['execution_note']='mixed recorded CPU/CUDA phases; source factory bindings and archived failures retained; no threshold/geometry/seed tuning'
(ROOT/'CONFIG_RESOLVED.json').write_text(json.dumps(resolved,indent=2))
print(json.dumps(dict(reports_written=7,main_records=36,control_records=2,A17_receiver_wins=wins,median_ratio=median,wall_s=wall)))
