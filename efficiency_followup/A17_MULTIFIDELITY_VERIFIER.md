# Anchor能否免去完整核验？有限覆盖成立，廉价拒绝没有实际收益

## 状态与输入边界

结论：**NO-GO，作为本实现的廉价核验策略。** 本报告只做保存数据上的误差校准与cheap-reject诊断，没有cheap-accept、没有训练网络，也没有改变original或新frequency策略的验收。

64-column richer reduced anchor 已用于 proposal 排名。对材料位移d，令z_A=J_A d、z_F=J_F d；拟研究仅由在线量给出的半径eps，使输出缺陷满足||z_F−z_A||≤eps。若这不等式成立，可以获得收益区间，但本实验的eps来自经验校准，不是确定性全波误差界。

先固定平滑voxel为development、壳体Gaussian为validation，再读取其标签。两种半径均取development最大norm ratio的1.1倍：

\[
\epsilon_{A}(d)=0.9960683840733512\,\|J_A d\|,
\qquad
\epsilon_d(d)=0.0898323733945177\,\|d\|.
\]

没有根据壳体结果收紧常数。材料范数采用原物理正交实坐标，测量范数采用原realification和data normalization。

## 避免为g_F(x)额外支付adjoint

原在线程序已支付u=r+J_Fx，但没有持有g_F(x)的完整向量。直接使用用户所给的−g_F(x)^Td半径公式会遗漏这项成本。可以改写成使用u及anchor输出的区间：

\[
\begin{aligned}
Q_{UB}={}&-u^Tz_A+\|u\|\epsilon
 -(\ell+\Lambda x)^Td
 -\tfrac12\max(0,\|z_A\|-\epsilon)^2
 -\tfrac12d^T\Lambda d,\\
Q_{LB}={}&-u^Tz_A-\|u\|\epsilon
 -(\ell+\Lambda x)^Td
 -\tfrac12(\|z_A\|+\epsilon)^2
 -\tfrac12d^T\Lambda d.
\end{aligned}
\]

这是Cauchy不等式及输出norm三角不等式的直接结果，**条件是半径确实覆盖输出误差**。这里u、d、prior项和z_A可在线获取；true z、true Q只用于离线检查。screen feature文件先写出，再读取离线标签，来源哈希有记录。

诊断只在Q_UB≤0时提出廉价拒绝。零是非负online tolerance的保守下界；不提前调用依赖Jd的原门槛。没有使用Q_LB进行廉价接受，所有部署正收益仍需完整Jd。

## 306个已付费finalists的结果

| 对象与来源 | 样本 | 两种半径及Q区间覆盖 | anchor半径拒绝 | displacement半径拒绝 |
|---|---:|---:|---:|---:|
| 平滑development nonlinear | 100 | 100/100 | 0 | 0 |
| 平滑development frozen | 48 | 48/48 | 0 | 1 |
| 壳体validation nonlinear | 106 | 106/106 | 0 | 0 |
| 壳体validation frozen | 52 | 52/52 | 0 | 0 |

唯一可拒绝的是一个平滑early/k4的原负收益候选；回顾性对应6照明RHS，**没有真正执行的节省**。两对象全部nonlinear finalists均未触发。区间很保守，finite coverage良好并没有转成工程收益。不能通过看完held-out数据再缩小eps来制造拒绝。

306行不是306个独立物理对象：同一状态共享L、B、anchor与条件路径；候选又已被原shortlist选择。发展对象的max-ratio覆盖主要来自校准方式，validation也只有一个对象。不能将158/158变成分布外保证。

## 结论与下一步边界

不把这个triage部署到推荐路径；它没有支持省Jd的实测收益。相对norm半径接近anchor输出norm，也说明“anchor排序有用”与“anchor足够精确地替代输出核验”是两个问题。

向A18交付实际支付的 true_z、anchor_z、online features、true Q、sample_id/source/timing，以及 reduced-endpoint fallback metadata。defect `(J_F−J_A)d` 由离线 sample_id join 后相减得到；当前没有独立 full-verifier fallback 实测标签。Jd仅占小部分增量时间，学习这个defect不会自动解决重复公共空间建设或line-search成本。本任务不训练它。

证据：`research/delegated/a17_eff_saved_profile/RADIUS_DIAGNOSTIC_PREDECLARATION.json`、`SAVED_OUTPUT_INTERVAL_DIAGNOSTIC.md`、`OUTPUT_INTERVAL_PREDECLARATION.json`、`output_interval_receipt.json`、`output_interval_screen_features.*`及`output_interval_pairs_offline.*`。旧radius报告中“缺g_F(x)所以未做部署区间”的限制，被本报告的已付费u形式仅在区间诊断层面补充；没有改动旧证据或声称免费完整梯度。
