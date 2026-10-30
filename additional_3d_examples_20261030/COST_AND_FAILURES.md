# 完整费用与真实运行故障

## 累计资源

- 实际受控计算wall：**8700.103922秒，2小时25分钟**；用户批准上限10800秒。
- full矩阵分解准入（账本states）：**3084**，上限4096。
- 复full-solver RHS列数：**63808**，上限150000。
- 主任务36条，追加控制2条，全部MAX_UPDATES且18次接受更新。没有训练、追加形状、改噪声或放宽数值门槛。

这是本独立补充研究的累计账本，不含旧论文和旧三维研究费用。导入前启动、网络传输、打包和离线绘图不属于受控求解wall；一次缺依赖的启动在建立Budget前失败，状态/RHS为0，其启动wall未单独计量。不能把此数字叫完整用户等待时间或纯GPU-active时间。

## 全部受控阶段

| phase | PID | 记录状态 | 占用wall秒 |
|---|---|---|---|
| validate | 70700 | COMPLETE | 1.284 |
| run | 71023 | ERROR | 2263.889 |
| XINAN_i9_continuation | 20044 | ERROR | 1029.262 |
| XINAN_4060_continuation | 2456 | ERROR | 678.763 |
| XINAN_4060_parallel | 19120 | ERROR | 1241.512 |
| XINAN_4060_parallel | 14252 | ERROR | 623.094 |
| XINAN_4060_parallel | 23388 | NATIVE_CRASH_C0000005 | 281.000 |
| XINAN_4060_serial_native_resume | 12264 | ERROR | 1110.934 |
| XINAN_4060_serial_native_resume | 13632 | ALL_PLANNED_TERMINAL | 953.422 |
| XINAN_GPU_serial_ordinary_gradient_control | 15612 | ALL_CONTROL_TERMINAL | 516.944 |


并行阶段按master进程占用计费，子轨迹wall不相加。23388的原生崩溃没有正常close；按系统事件的进程生存时间取上整再加1秒，保守计281秒。其他正常关闭阶段保留原记录。中断tail和接续重新构建当前物理状态的费用未扣除。

## 每条主轨迹的实际费用

| 目标 | 方法 | full分解准入 | 复full RHS | 子轨迹wall秒 | 模型counter范围 |
|---|---|---|---|---|---|
| 立体数字3 | full_gn | 75 | 3204 | 158.620 | whole trajectory |
| 立体数字3 | receiver | 84 | 1224 | 178.345 | whole trajectory |
| 立体数字3 | projected_internal | 90 | 1296 | 208.125 | whole trajectory |
| 立体数字3 | fourier | 88 | 1272 | 178.045 | whole trajectory |
| 立体数字3 | random | 99 | 1404 | 203.879 | whole trajectory |
| 立体数字3 | exchange | 36 | 1512 | 161.911 | whole trajectory |
| 字母E | full_gn | 67 | 3108 | 154.943 | whole trajectory |
| 字母E | receiver | 99 | 1404 | 176.733 | whole trajectory |
| 字母E | projected_internal | 83 | 1212 | 185.650 | whole trajectory |
| 字母E | fourier | 90 | 1296 | 203.045 | whole trajectory |
| 字母E | random | 89 | 1284 | 186.032 | whole trajectory |
| 字母E | exchange | 74 | 2040 | 256.447 | whole trajectory |
| 单环 | full_gn | 88 | 3360 | 404.383 | whole trajectory |
| 单环 | receiver | 121 | 1668 | 440.678 | whole trajectory |
| 单环 | projected_internal | 89 | 1284 | 1028.847 | whole trajectory |
| 单环 | fourier | 91 | 1308 | 340.172 | whole trajectory |
| 单环 | random | 90 | 1296 | 336.273 | whole trajectory |
| 单环 | exchange | 51 | 1620 | 952.337 | whole trajectory |
| 双环 | full_gn | 87 | 3348 | 241.801 | whole trajectory |
| 双环 | receiver | 66 | 1008 | 172.853 | whole trajectory |
| 双环 | projected_internal | 84 | 1224 | 206.687 | whole trajectory |
| 双环 | fourier | 90 | 1296 | 313.269 | whole trajectory |
| 双环 | random | 89 | 1284 | 287.475 | whole trajectory |
| 双环 | exchange | 35 | 1524 | 421.362 | whole trajectory |
| 三轴交叉杆 | full_gn | 92 | 3524 | 223.588 | continuation_only |
| 三轴交叉杆 | receiver | 56 | 888 | 133.336 | whole trajectory |
| 三轴交叉杆 | projected_internal | 97 | 1380 | 233.705 | whole trajectory |
| 三轴交叉杆 | fourier | 94 | 1344 | 211.539 | whole trajectory |
| 三轴交叉杆 | random | 89 | 1284 | 238.720 | whole trajectory |
| 三轴交叉杆 | exchange | 54 | 1992 | 387.462 | continuation_only |
| 偏心核壳 | full_gn | 80 | 3264 | 187.143 | whole trajectory |
| 偏心核壳 | receiver | 66 | 1008 | 152.475 | whole trajectory |
| 偏心核壳 | projected_internal | 68 | 1032 | 174.131 | whole trajectory |
| 偏心核壳 | fourier | 90 | 1296 | 206.508 | whole trajectory |
| 偏心核壳 | random | 88 | 1272 | 201.752 | whole trajectory |
| 偏心核壳 | exchange | 76 | 2208 | 268.898 | continuation_only |


前两列物理计数从全局原始账本的case/policy stage汇总，包含原前缀、失败尾部与接续。模型counter若标continuation_only只覆盖新进程，不能据它认定原前缀免费。中断轨迹wall含保留前缀的原记录或保守时间估计。墙钟跨Mac、i9、CUDA CPU端点/CUDA端点以及串并行阶段，**不可直接构造算法加速比**。运行中还观测到其他CPU任务；没有干预它们。

普通梯度两条各有86次full分解准入、1248个复RHS。数字3wall 258.749秒，单环257.757秒。它们的更低材料误差，不自动等于更低总计算成本。

## 真正故障与交接应分开

1. **首次远端导入失败**：缺threadpoolctl，尚未建立Budget，无状态/RHS；复制本机已有纯Python模块后启动，不安装新框架。原日志保留。
2. **PID23388原生崩溃**：Windows事件记录python310.dll access violation 0xc0000005；未找到Python traceback。当时是双线程任务进程。根因未确定，不能仅凭并发时序归因于CUDA或线程安全。保留全部完成轨迹，从cross3d/exchange的chi09和full_gn的chi03接续，未重复已接受更新；后续单GPU作业内仍保留批量多RHS。
3. **PID12264日志命名空间异常**：offset_shell/exchange在4次接受更新后，clean中的np.ndarray访问报告np为builtin function/method。静态阅读没有找到正常的显式覆盖路径；根因和数值影响范围未确定，不能宣称只影响日志。原final/error/partial完整存档；换新进程从chi04接续，未改公式、候选或门槛。未提交的交换轮重新计算并收费。
4. 若phase状态ERROR来自REFUSE_EXISTING_RESULT及HANDOFF_RESERVATION，是在下一条未开始任务处安全退出的硬件/进程交接；它不是该目标的数学失败。原状态保留，不把它改成PASS。

这些恢复没有增加验证测试或bench，也没有给运行异常的旧前缀补独立数值背书。新研究结果的这一限制与原A17论文证据分别处理。静态故障说明附在[LOGGING_NAMESPACE_STATIC_NOTE.md](LOGGING_NAMESPACE_STATIC_NOTE.md)。

## 硬件实际使用

Mac CPU阶段和i9 CPU阶段后，完整DDA求解切换到原CUDA complex128路径；之后压缩材料端点也切换到CUDA complex128/float64，并批量处理同轮候选。初始i9纯CPU阶段使用8个数值线程；后续CUDA阶段为4线程，后期进程affinity限定为已识别的P-core逻辑处理器。这些设置不是最初全部阶段都已采用。双轨并发发生原生退出后，改为一个GPU作业；没有取消作业内批量。

运行快照曾观察到GPU约69%及100%利用率，也有CPU构造/投影/日志期间的0%快照。没有测量完整GPU-active积分，不能把峰值或瞬时空闲当作整体利用率或性能结论。
