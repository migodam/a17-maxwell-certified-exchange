# 一个电流模式何时值得为材料更新保留？

**四类三维 vector-Maxwell 对象、相同电流维数下，最多三次条件交换在35/36个冻结单元改善完整GN材料步；对象汇总风险比中位数为0.348，差距降低约65.2%。** 这个正面结果把物理机制推进成了可执行的局部电流设计。它的贡献在于同预算的更新保真度，以及能解释、能检查的选择条件。

## 1. 问题是怎样形成的

电磁逆散射（electromagnetic inverse scattering，EIS）用散射场推断物体材料。材料变化先改变内部电流，电流经过多次相互激发，最后才到达接收器。Gaussian/N-Port最初提供的是场和电流压缩平台，但材料反演真正需要的问题更具体：一个电流方向，经过怎样的物理作用，才会影响当前材料步？

SOM按接收Green算子的奇异结构分析current observability；Twofold SOM也研究内部传播。本文从这些已有认识出发，追踪一个指定current-space change，而不是构造一个“忽略物理”的对手。真实材料导数写作 J=S L⁻¹ B：B是局部总场和真实极化率导数形成的材料注入，L⁻¹包含多重散射，S是接收映射。材料度量、实化和所有照明的共享变量固定，避免参数缩放制造假现象。

保留空间消元说明：新增电流不只直接辐射，还激发原有电流并经内部反馈返回。导数改动因此分成feedback-dressed observation、Schur coupling与material injection。对应两组材料pullback完整包含材料步差。这回答“改动会影响哪里”，还没有回答“当前误差最需要哪里”。

## 2. 为什么结构修复还不够

历史实验中，paired repair优于指定direct current enlargement：平滑72/72，尖锐36/36。等材料秩Krylov又在全部72/72和36/36更强；paid paired warm route时间是PCG的1.416倍，完整RHS从66增至342。这不是对机制的反证，而是对把结构空间当成最佳误差修正空间的限制。

同样，改变电流模型并不是增加独立观测。曲率差包含与旧Jacobian的交叉项，可以不定。一个模型声称的响应更强，不代表它更忠实，也不代表它在当前残差下生成的步更好。

因此，研究收缩为一个可以真正做决定的动作：已有可靠receiver基底，只换掉少量方向，问新模型产生的实际材料步是否更好。无需先推倒整个基底，也不预设必须开发更快solver。

## 3. 条件交换怎样执行

从receiver spectral backbone出发。每一轮先选一个要移出的方向，再从固定的家庭均衡十二候选短池提出incoming方向。每个child endpoint是实际复电流空间，按完整Galerkin关系求自己的材料步。richer reduced anchor给代理收益，在线不读取完整最优步或真值。

真实位移是 d=s_new−s_old，不是只用旧Hessian给的线性化方向。缓存 Jx 后，每个finalist只计算完整方向场 Jd：
Q=−(r+Jx)ᵀJd−(ell+Lambda x)ᵀd−0.5(||Jd||²+dᵀLambda d)。

每轮最多检查两个finalist，通过冻结完整收益规则才接受，最多三个动作。接受后更新空间、当前材料步、缓存输出和条件耦合；完整目标和richer anchor保持固定。这叫conditional exchange，因为同一个incoming方向在不同保留核心里的作用不同。

完整物理检查没有免费：六次illumination对应每个方向六份current RHS。它的好处是将完整物理信息集中花在少量真实动作上，不需要新full adjoint来得到标量gain。

## 4. 三维证据的范围与正面结果

四个固定对象为平滑Gaussian、近接触Gaussian、分层voxel和非对称voxel。每个对象初始、中期、晚期三个保存状态，在 k=4、8、16 测试，共36单元、180条policy记录。电流k是所有六次照明共享的实际复列数。Gaussian物理实切向为54维，voxel为3456维。

完整forward residual、总场、真实material injection、ell、prior和LM不变，只改tangent current。两个开发对象先过门槛后，按冻结规则扩展另两个对象；没有按结果换对象或提高通过阈值。

|几何 / 材料表示|Local teacher风险比|Verified exchange风险比|两条路径收益取得比例|
|---|---:|---:|---:|
|平滑 / Gaussian|0.0984|0.1904|89.8%|
|近接触 / Gaussian|0.2016|0.6538|43.4%|
|分层 / voxel|0.1520|0.2776|85.2%|
|非对称 / voxel|0.1628|0.4181|69.5%|

风险是完整GN quadratic gap，即约化材料步与同一完整最优步之间的曲率误差。每个对象九单元先求和再相除；四对象比值中位数0.34785，相当于下降65.2%。这不是最终材料图像误差。35/36有可分辨改善，另一个不变，36/36没有可分辨退化。

Teacher枚举完整声明一换一邻域，但路径最多三个局部动作，不是全局oracle。Verified接受75个动作，额外finalist tangent RHS为1164，初始化和投影另外收费。≤k终态全部仍用满k，没有rank saving。

## 5. 最有价值的诊断：候选覆盖比评分误差更直接

在各对象的九个共同receiver起点，先把单次收益求和再相除，完整字典anchor top-one对象级捕获率为98.1–99.6%。这并不表示每个checkpoint都正确；一个平滑中期k16的强制top-one仍选到负gain，部署复核会拒绝它。

在线十二候选短池本身包含的最好机会，平滑为96.9%、近接触35.3%、分层92.0%、非对称65.9%。近接触对象就算把池内所有得分算准，仍拿不到池外的好方向。这个分解比笼统说“surrogate没有达到oracle”更可操作：优先改进便宜的incoming覆盖，而不是默认必须换复杂网络。

表中的多步路径收益比例还有条件搜索与路径分歧，不是纯score regret。评分、候选覆盖、路径搜索是三个不同的损失来源。

## 6. GN信息层：解释哪些材料方向改变，但不替代收益

固定物理尺度 P=10⁻⁵I，数据曲率为 I=JᵀJ。Information volume和effective dimension衡量尺度归一化响应；数值LM单列。这里没有实测噪声协方差支持的互信息解释。

模型变化 D 同时改变：
ΔI=J_oᵀD+DᵀJ_o+DᵀD，
Δh=Dᵀr。
实际材料步差 d=−H_n⁻¹(Δh+ΔI s_o)。曲率与offset共同决定它，不能只按谱排序。

一个真实近接触rank4交换，信息体积增加6.3127、有效维数增加1.8613、归一化曲率fidelity略有改善，完整gain却为−5.5754×10⁻⁵。保存的状态、基底、步与方向数组相互绑定。它直接展示：信息幅度和一个保真标量都不足以决定当前材料收益。该例没有孤立证明offset是唯一主因。

共同rank14核心另有两singleton gain为−3.2391×10⁻⁸、−5.8389×10⁻⁸，联合rank16 gain为+3.4787×10⁻⁷。540个固定pair测试中5个严格见证，说明条件耦合确实存在；不是所有greedy方法失败的概率或同秩swap陷阱。

## 7. 有限位移与一阶评价的受控比较

旧finite/first-order比值0.1171混有方向构造和terminal swap，不能单独归因curvature。

新的共候选诊断固定起点和十二候选，交叉 v/d 与linear/quadratic score，统一执行真实d。强制top-one、不含no-op时，完整线性评价在29/36选负gain，完整二次评价降为1/36。它证实在该共同候选诊断中，真实有限位移的二次惩罚不可省略。部署另有no-op与验收，没有把强制诊断当算法失败率。

## 8. 全部费用与效率

17个远端attempt均有终态，含两次失败，作业占用4889.515秒，即81.49分钟；包含CPU、teacher、诊断、计时重复和预热，不是GPU持续active。显存最大1872MiB，无OOM。

相同CUDA FP64物理算子的五次轮换计时，conditional warm selection：
Gaussian k4/8/16为0.594/1.418/2.690秒；
voxel为0.891/1.153/3.400秒。
一次同池定向决定更便宜，固定one-pass也更便宜，但检查与搜索语义不同。新方法支付费用换取完整更新保真度，没有声称receiver不动更慢。

公共physical setup约5.43/5.73秒；历史anchor/pool、离线评价、拒绝、fallback及预热分别入账。独立cold end-to-end时间未知。已知all-failed分支可能漏一个阶段计时，但全部job wall仍收费，本次timing没有触发。

## 9. 学习、停止和重建的清楚裁决

两个小模型各三个seed，按对象train/validation/test分组。MLP heldout top-two收益捕获98.85–99.16%，group-mean75.99–98.03%，都低于anchor99.885%。达到70%retention，未证明全费用减半；六run保留，不把NN作为论文贡献。

严格output-error radius尚未取得。22单元ABSTAIN_UNRESOLVED，14到move budget，没有全邻域certificate。Gaussian六个完整参考在固定nu≤1阈值下weakdim为0；voxel只诊断共同十六维投影，完整空间weak/null方向仍未定。没有启动Flow/prior。

无约束alpha=1真值诊断相对当前材料19变好、17变差、28违反约束。这不是receiver材料端点对比，也没有经过line search。新nonlinear reconstruction按冻结合同未运行。

## 10. 论文现在可以有力写什么

可以直接写：可靠receiver基底附近，少量材料相关、经过完整gain复核的条件交换，明显改善同current budget的完整材料更新；物理feedback与paired support给出动作作用的解释；评分与候选域损失可量化分开；信息增长并不保证任务收益，真实Maxwell中有可重算见证。

历史宽域FAIL_KEEP_MECHANISM继续保留，sharp恢复限制、coarse sphere Mie21.40%、paired1.416×及Krylov更强都保留。独立中等对比球32³的场0.252%、均匀导数0.731%验证也只覆盖相应球体与均匀方向。

文章定位为mechanism + conditional current design。标准Schur、Woodbury、GN/Fisher和ROM/Krylov不重新包装为创新；SOM/TSOM部分primary full text仍OPEN，不能写first-ever。未来证据要解决低成本proposal覆盖、合法output bounds、噪声及有约束nonlinear transfer；不影响这里已经得到的局部设计正结果。

