# A17 conditional tiny proposer — frozen preparation contract

状态：PREPARED_TOY_ONLY。未读真实 teacher labels、未训练真实模型；必须收到父线程明确四对象 NN gate PASS 授权后才能启动。该门槛不等于原两对象 expansion gate。固定设置见 FROZEN_CONFIG.json，哈希已绑定 TOY_CHECKS.json；若未来要改架构/特征/epochs，应先新版本预注册，不能根据 heldout test 改。

训练对象2007/2012；validation2001；heldout test2014。所有 phase/k/候选属于对象所在组，不做 candidate split。validation仅报告，不选模型、不early stop；test仅在模型训练与保存后计指标。六个架构/种子组合全部保留，不选最好的 seed。单个 heldout 对象只支持有限的对象外案例证据，不能以候选数量宣称普遍泛化。

真实读取接口要求 dataset manifest 锁定每个完整 receiver checkpoint 的源JSONL SHA256、行数、feasible数、result COMPLETED、same base_U_hash、round0/policy身份。只有原 declared complete1swap 的 OK feasible候选参加排名；所有失败不可行行由原输入manifest/trace保留。真实入口在打开manifest或labels前检查授权。全对象均必须出现；manifest给定split必须与冻结split一致。

允许特征共7个加7个missing flags：Qhat / 同 checkpoint 在线 mean|Qhat|（下限1e-12）、signed log1p(receiver_score)、log1p(step_difference_norm)、log1p(child_condition.condition)、child_condition.relative、log1p(information_trace)、log1p(k)。只访问这些字段；Qtrue只作训练target/离线评价。actual atom IDs只定义group关系，不作为数值特征。缺feature按0与独立missing flag表示；归一化仅train candidate mean/std，std<1e-12按1；不读取test统计、fullrisk或full optimal来确定尺度。

Qhat需要anchor child评价，step norm/current稳定性/trace可能需要child solves；此名单合法不代表便宜或免费。保存的trace字段已有费用必须在真实数据输入cost记录关联。信息trace作为已付cheap候选描述，不替代完整物理验收。完整Q/risk/optimal/truth/teacherbest/full与projected fidelity/evaluator reference明确排除。

两模型：MLP `14→32→16→1` ReLU；group-mean模型先`14→32`，分别汇总同incoming与同outgoing的hidden mean，concat自身+两mean `96→16→1`。按declared single-swap IDs分组，scatter/index-add方式不构造dense allpairs。CPU单线程、确定性操作、FP64参数/训练；FP32仅做toy前向等变检查，不把两精度混为相同训练结果。

训练：固定100 epochs、Adam lr1e-3、weight decay1e-4；按seed随机排列checkpoint，每checkpoint一次mean candidate MSE更新；label target为 Qtrue / 在线Qhat尺度。每checkpoint同等更新数，训练normalization按train候选加权，不假装全部候选独立。seeds 20261002/3/4；不根据validation/test调整。

输出是同receiver checkpoint一次 top2 priorities，不跑多轮交换/重建。priority tie按原move_index。top2不按预测正负裁剪。离线取得两label后评估最好的finalist或no-op，标OFFLINE_ORACLE_AMONG_TOP2_PLUS_NO_OP；此动作不等于fresh physical acceptance。另保留top1原signed gain、signed capture和未经no-op的regret。主capture=best(top2,no-op)/best(full neighborhood,no-op)，regret为两者差；不人工截断。teacher无正机会时capture=null+NO_POSITIVE_TEACHER_OPPORTUNITY。科学阈值/安全验收仍须原registered tau与独立复核，网络不提供certificate。

费用：记录CPU feature数组构造、inference、训练wall；前者只计算读入字段转换/归一化，不含原全候选anchor/child采集。fresh physical验证/all候选物理features/all-in speedup均null+NOT_MEASURED。toy timing与toy labels不用于科学成本或NN gate。

最少matched计费方案（待父线程授权执行）：选一个Gaussian和一个voxel的同receiver/basehash、同完整候选集；baseline与两模型共享相同state/anchor/init、feature permission与top2独立物理验收，记录冷setup、整候选cheap features（含Qhat/child solve/stability/info_trace）、normalization/grouping、inference、每finalist full Jd、Jx init、accepted/no-op、峰值内存、完整occupied interval。单列已支付teacher/训练与共享费用；分别报cold与warm，勿累加重叠阶段。至少3次固定repeat以显示计时变动；跨checkpoint候选集相同，baseline不使用昂贵full oracle。两例仅证明局部matched费用，最终≥2x总成本需对应heldout测试工作负载全费用覆盖，不能从当前offline labels或inference时间推出。

入口 `tiny_proposer.py --manifest DATASET.json --gate-authorization OWNER_AUTH.json --out <本目录新目录>`。授权需 four_object_nn_gate=PASS、owner_authorized_real_training=true、objects=[2001,2007,2012,2014]、evidence_sha256非空。这里提供模板，绝不生成授权PASS。输出每架构/seed模型state_dict与receipt、normalization、six-run index、授权/manifest绑定。模型.py只导入CPU torch/numpy，无remote/Maxwell接口。
