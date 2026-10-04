# 定向切向后端：原实现已有多右端求解，另行检验精确几何缓存

## 后端与一致性

材料位移采用实数物理正交坐标。对 B 个位移及 P=6 个照明（illumination），原完整切向路径构造形状 `(P,3N,B)` 的极化率导数注入，并打包成 `(3N,P×B)`；N是物理网格点数。一次 CUDA complex128 `lu_solve` 使用同一固定状态（frozen state）的已支付 LU。结果回传为全部物理电流（current），再用相同 G_S 观测和复测量实化（realification）形成 Jd。这是共享分解的多右端求解（multi-RHS block solve）。

因此两 finalist 的核验本来就是12 RHS 的一次 block solve，而不是重新分解12次。endpoint 的 Gaussian 小材料系统在CPU上求解，不意味着 full Maxwell solve 没有CUDA。这里不引入新的后端、精度或网格。

新的专门控制对原 block 与逐位移调用作三次轮换顺序比较，FP64、输入、LU、输出和验收均固定。每对象两路线合计支付72 tangent RHS；这些诊断费用单独计入总预算，不用它们声称节省物理作用。

| 对象 | block 中位时间 | 两个单方向调用中位时间 | block/单方向时间 | RHS/每路线 | 新LU/核验 |
|---|---:|---:|---:|---:|---:|
| 平滑 voxel | 0.083642 s | 0.075921 s | 1.102 | 12 | 0 |
| 壳体 Gaussian | 0.083873 s | 0.076341 s | 1.099 | 12 | 0 |

**没有测得新的 block wall-time 优势。** block将2次solve调用合并成1次，但这个小控制中略慢，不能择取最快重复制造收益。完整原生产路径已经使用block，保留它；候选数量减少与kernel速度改变分开报告。

profile 中独立拆分的原 J 作用与实际原输出的相对差为0。cache/block控制同时核对观察输出、方向收益Q、在线数值门槛、eligible flags和选择；没有 deterministic Maxwell output-error radius，因此这些是高精度一致性检查。

## 精确 geometry cache：另一条独立优化

缓存对象仅为原 `domain_bank` 的 V/meta。它由背景 geometry/G_D、seed和请求列数决定，不存储材料状态 L、B、Jd、候选评分、anchor或旧endpoint。每次状态仍使用自己的 injection、全波反馈和 material step。

| 对象 | 独立原 bank 获取 | 真实 cache hit，含哈希/复制 | 单 bank 速度比 | actual acquisitions/hits |
|---|---:|---:|---:|---:|
| 平滑 voxel | 4.668234 s | 0.656086 s | 7.115 | 1/1 |
| 壳体 Gaussian | 4.654615 s | 0.652324 s | 7.135 | 1/1 |

两个对象在不同材料状态上的 V 与 metadata 均逐字节一致。cache key 对源码/provider、可执行语义、几何及G_D作绑定；命中仍包含大数组指纹和复制费用，不是零成本。当前 wrapper 对 anchor/pool 调用都检查，实际整体速度须由完整重建测量。

## 保留的失败版本

V1虽输出正确，却出现 `acquisitions=2,hits=0`。原因是用序列化 code object 作key时出现运行前后指纹漂移；名为 `first_hit_check` 的阶段名称不足以证明缓存命中。该版本没有通过准入，保留31.250 s外部occupation及原源码/结果。

V2独立命名，使用稳定的可执行语义序列及source/provider身份；新增实际MISS/HIT断言。两对象真实观察到1次MISS、1次HIT，才准入完整重建。没有回写原A17代码或覆盖V1。

## 裁决

- 真 multi-RHS：**已成立，原实现已有**；本任务不以此为新贡献。
- 新block相对逐列的测量收益：**NO-GO**，没有可支持的加速。
- 精确 geometry-only cache：**通过两个实际物理对象的一致性准入**；端到端效果由独立配对重建决定。
- 整状态位移SVD压缩：见位移秩审计，未加入部署。

证据：`research/CACHE_AND_BLOCK_OWNER_ADMISSION_V2.json`；`remote_snapshots/cache_block_2002_complete_v1/`、`cache_block_2002_complete_v2/`、`cache_block_2009_complete_v2/`；`code/geometry_cache_v2.py` SHA256 `022362536c6f02de673af5895cb3476dd88b5d3fc8d69c6adcf1ba53af9c6295`；`code/verify_cache_block_v2.py` SHA256 `dff085bb9df6af0739acc5e6ef28b0e7f251193b085758d3d39e38296ae93bc5`。
