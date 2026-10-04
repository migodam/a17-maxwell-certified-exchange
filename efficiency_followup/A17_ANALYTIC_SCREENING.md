# 用收益上界提前拒绝候选：恒等式成立，受测数据没有成本优势

结论：**NO-GO，作为当前部署优化不推荐。** 下述代数界成立；问题在于所需梯度不是当前程序已经支付的梯度，且可拒绝候选很少。没有修改原验收规则。

## 为什么这个上界安全

固定一个材料状态及完整线性化，材料系数为实数。令当前约化步为 x、候选位移为 d，完整正则二次函数的梯度为 g_F(x)。实际收益为

\[
Q=-g_F(x)^Td-\tfrac12\|J_Fd\|^2-\tfrac12d^T\Lambda d.
\]

因为切向场能量非负，

\[
Q\le Q_{\rm upper0}=-g_F(x)^Td-\tfrac12d^T\Lambda d.
\]

在精确算术下，只有实际验收门槛已因果可得，或持有预声明、非负的保守门槛下界时，才可据此拒绝；本保存诊断使用下界0。数值实现仍须为上界运算留出舍入余量；不能把数值显著性门槛称为 Maxwell 输出误差证书。

## 付费梯度的区别

完整重建每个 outer iteration 已算的是 g_F(0)=J_F^Tr+ell。候选核验发生在已求得的约化材料步 x 处，通常 x 不为零。因此

\[
g_F(x)=g_F(0)+J_F^TJ_Fx+\Lambda x.
\]

Jx 已为定向核验支付，但 J_F^T(J_Fx) 没有免费获得。每个接受后的新 x 又改变该梯度。额外 adjoint 必须计费。保存的 g_F(0) 不可替代 g_F(x)。

已付费量能给出一个更松的严格界：由 Jx 与未知 Jd 的 Cauchy 不等式及完成平方，

\[
Q\le-g_F(0)^Td-(\Lambda x)^Td-\tfrac12d^T\Lambda d+\tfrac12\|J_Fx\|^2.
\]

这个松界不需要新增 adjoint，但在下表的全部 nonlinear finalist 上均未触发。

## 保存数组的逐候选审计

| 对象/数据范围 | finalist 数 | 假定 g_F(x) 已有时上界可拒绝 | 错拒正收益 | 真正实现节省的 RHS |
|---|---:|---:|---:|---:|
| 平滑 voxel，18 次非线性更新 | 100 | 0 | 0 | 0 |
| 壳体 Gaussian，18 次非线性更新 | 106 | 3 | 0 | 0 |
| 近接触 Gaussian，补充保存数据 | 106 | 5 | 0 | 0 |
| 平滑 voxel，9 个冻结状态/秩单元 | 48 | 1 | 0 | 0 |
| 壳体 Gaussian，9 个冻结状态/秩单元 | 52 | 5 | 0 | 0 |

这里的上界由保存的 Jd 重建，是回顾性诊断，不能当作一个部署时已免去 Jd 的实验。表内使用零收益门槛；原在线门槛的部分尺度含有未计算的Jd，不能提前假定已知。零是非负验收门槛的保守下界，因此这里只计算可证明不可能取得正收益的子集，不代表原数值门槛下的全部潜在拒绝。零错拒确认了保存量的关系；并没有证明任何免物理作用的梯度获取办法。各方向对应六个照明 RHS。即使壳体那三个方向可避开，额外梯度作用也必须与18次及接受后的刷新相比。

## 影响

不把这个筛选加入推荐路径。它不改变原 A17 科学结果。若另一个应用确实已经持有 g_F(x)，这个上界可以使用；本实验没有这样的共同沉没成本。

证据：项目根目录 research/delegated/a17_eff_saved_profile/SAVED_SCREENING_AVAILABILITY.md、aggregate_findings.json、screening_finalists.csv、frozen_retrospective_screening.csv。每个源数组和运行模块均在 evidence_manifest.csv/runtime_import_bindings.csv 绑定。这里只进行保存数组分析，没有新增 full-wave action。
