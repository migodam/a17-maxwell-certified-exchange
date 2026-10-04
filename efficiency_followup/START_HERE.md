# A17 verified exchange：效率后续证据

**结论CONDITIONAL。** 推荐每轮只核验第一名、最多三次交换、精确几何缓存。在两个规定对象中，原交换重建时间从581→425秒、593→536秒，质量保住；相对unchanged receiver仍1.96×/1.32×，没有普遍加速claim。

先读 [最终报告](A17_EFFICIENT_VERIFICATION_FINAL.md)，再看 [收益—费用曲线](A17_EFFICIENCY_PARETO.md) 和 [真实瓶颈](A17_EFFICIENCY_PROFILE.md)。最重要的新发现是Jd只占原额外时间约1%–4%，主要直接增量来自反复bank/anchor/workspace构建。最多两次交换虽保住约99%冻结收益，仍损失平滑重建质量，完整数据保留这一失败。

## 十一份报告

1. [Runtime profile](A17_EFFICIENCY_PROFILE.md)
2. [解析拒绝上界](A17_ANALYTIC_SCREENING.md)
3. [位移秩](A17_DISPLACEMENT_RANK_AUDIT.md)
4. [Block与精确缓存](A17_BLOCK_SOLVE_AUDIT.md)
5. [核验频率](A17_VERIFICATION_FREQUENCY_ABLATION.md)
6. [跨状态空间复用](A17_STATE_REUSE.md)
7. [Anchor经验半径](A17_MULTIFIDELITY_VERIFIER.md)
8. [Verification ROM：未运行原因](A17_VERIFICATION_ROM.md)
9. [Pareto曲线](A17_EFFICIENCY_PARETO.md)
10. [A18数据接口](A17_TO_A18_EFFICIENCY_INTERFACE.md)
11. [最终裁决](A17_EFFICIENT_VERIFICATION_FINAL.md)

## 数值与来源审阅

[覆盖](A17_EFFICIENCY_COVERAGE_FINAL.md)、[完整费用](A17_EFFICIENCY_COST_FINAL.md)、[失败/边界](A17_EFFICIENCY_FAILURES_AND_LIMITS.md)、[claim](A17_EFFICIENCY_CLAIM_LEDGER.md)、[复核与重画](A17_EFFICIENCY_REPRODUCTION.md)、[图的检查](A17_EFFICIENCY_VISUAL_QA.md)、[交付状态](A17_EFFICIENCY_READINESS.md)。

两对象、34完整轨迹、90新冻结endpoint；三次重复只作计时。原A17的36 cells、holdout、噪声与机制witness保持原结论。此目录不训练NN、不创建selector、不发明新的solver。

`analysis/public_reanalysis_v1`包含透明的来源路径投影、源/配置哈希、数值表和完整schema审计表，可用于重画；公开相对来源名不意味着原始大数组已经随包分发。`A18_INTERFACE`包含历史finalists的anchor/full Jd与online features，不代表训练成功。

紧凑包为evidence/reanalysis，不是self-contained GPU replay。42 external vendor与17旧项目源仅有锁定身份；运行完整版需要私有source/input。不得从包能解压、图能重画推断完整物理重放。
