# 在线数值阈值回放

**PASS：原 36 个单元的交换动作与科学结果保持不变。**

本检查只改变接受动作的 numerical significance tolerance。原来的候选字典、receiver 起点、64-column reduced anchor、12-incoming pool、排序、top-two finalist 和三次交换上限均未改变。所有 finalist 都重新执行真实 FP64 Maxwell 方向作用；没有用已存收益的符号替代物理回放。

完整参考最优步仍用于离线评价，不能被送入新的在线接受函数。新的阈值由当前残差、已支付的 Jx/Jd、实际材料位移、正则项和 FP64 运算尺度计算。这是计算结果的显著性阈值，不是完整 Maxwell 输出误差证书。

| 回放量 | 结果 |
|---|---:|
| 已审计 state–budget 单元 | 36/36 |
| 原 / 新接受动作 | 75 / 75 |
| 最终 selected IDs、实际秩、停止原因 | 全部一致 |
| 每轮及终态材料步最大绝对差 | 0 |
| 每轮及终态 Jx 最大绝对差 | 0 |
| receiver / final GN gap 最大差 | 0 |
| finalist Qhat 最大差 | 1.30104e-18 |
| 全候选 reduced score 最大差 | 2.77556e-17 |
| 最大完整参考真残差 | 8.85831e-12 |

微小评分差没有改变排序或动作。原来的 35/36 improved、36/36 nonworse，以及对象汇总中位比值 0.347848 均保留。

来源：`analysis/replay_audit_v2/` 的逐单元比较、`analysis/replay_complete_owner_array_check.json`、`remote_snapshots/replay_complete/` 中已闭合的执行账本、真实 runtime imports 和数组。Windows 的 `__main__` 与 `__mp_main__` 是同一文件、同一哈希的模块别名；审计按唯一文件身份核对，并保留两个原始条目。

三个部署适配失败保留：保护系统进程的名称读取、远端缺少本地专用审计脚本，以及将物理 pool 的哈希与其压缩坐标哈希混比。前两项阻止实现 Gate；第三项在生成候选决策前终止。分别通过独立 Windows CIM 身份、精确远端 13-file source epoch、24 个历史缓存/receipt 的文件哈希绑定解决。旧失败版本和费用均保留；这些更改没有改变旧研究源码或选择规则。

最终验证配置在看到新对象结果之前封存：`configs/A17_FINAL_VALIDATION_LOCK.json`，SHA256 `a214e38954be1438a7b3e5a84cc6ea595d07544ceff652d6fd84995765a00ae5`，封存时间 2026-10-02 10:18:16 UTC。随后只允许冻结的 primary/noise 验证，不能按结果调整规则。
