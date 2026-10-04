# 候选位移的秩：大部分状态没有显著压缩余量

结论：**NO-GO，作为主要工程优化。** 同状态线性复用的代数正确，但实际 finalist 位移大多接近满列秩；跨状态的低秩不能用来复用已经改变的 Jacobian。

## 可用的精确线性复用

对同一个 frozen Jacobian，将已经可取得的候选位移组成 D=[d_1,...,d_p]。若 D=Q_DR 且分解无截断误差，计算 J_FQ_D 后可线性重构 J_FD。丢弃非零奇异方向是一种近似，不能仅凭“很小”称为精确复用。

材料位移按原物理正交度量实化。Voxel 使用 sqrt(volume) 加权的实/虚分量；Gaussian 使用保存的 Q_T，不重新生成或任意缩放坐标。保存的 coefficient fingerprint 和物理展开均已核对。

## 非线性保存数据

| 对象 | 18 状态总位移列 | 单状态数值秩分布（相对阈值1e-8） |
|---|---:|---|
| 平滑 voxel | 100 | rank6：12状态；rank5：2；rank4：2；rank3：1；rank2：1 |
| 壳体 Gaussian | 106 | rank6：13状态；rank5：3；rank4：2 |
| 近接触 Gaussian（补充） | 106 | rank6：13状态；rank5：4；rank3：1 |

同轮通常仅两个 finalist。完整单状态六列的回顾性分解也不能预先免费获得：后续位移依赖前一次接受结果。因此即便事后 rank 变小，也必须满足因果获取顺序才能实现收费节省。

壳体所有106列的跨状态秩是54，等于其材料实维度；平滑跨状态 rank 为95/95/98（相对1e-8/1e-10/1e-12）。这些几何结果不许可把旧状态的 Jd 作为新状态的 Jd。材料状态改变后，极化率导数、总场和全波 L 均改变。

## 冻结数据与实际作用核对

两对象 early/middle/late、k=4/8/16 的逐单元三个阈值及奇异谱保存在 frozen_displacement_ranks.csv/json。少数早期单元可少一列；多数中晚期仍为5–6列，没有预期的六列压到二三列的广泛收益。

新的平滑 middle/k8 实际六列有 rank6，最小奇异值约0.00282，最大约0.03745，FP64 QR阈值约2.87e-14。该控制不截断，也没有减少 RHS。原两个 finalist 的 block/逐列作用在专门的 block 审计中比较。

## 采用的处理

不向生产路径加入整状态 SVD/QR压缩。原 batch 已经共享 LU。继续保留精确的同状态线性复用接口和全部秩证据，作为将来候选更多时的诊断；不为现在最多两列增加构建成本。

证据：research/delegated/a17_eff_saved_profile/SAVED_DISPLACEMENT_RANK_AUDIT.md、displacement_ranks.csv/json、frozen_displacement_ranks.csv/json、gauge_checks.csv；EFFICIENCY_R1/remote_snapshots/cache_block_2002_complete_v1 的 verification_result.json。跨状态合并只是离线诊断。
