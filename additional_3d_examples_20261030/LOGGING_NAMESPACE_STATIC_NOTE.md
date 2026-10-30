# helpers np命名空间异常：只读静态说明

范围：本地当前 `original_exchange_helpers.py`、其isolated loader/controller、声明vendor依赖，以及新study的run_comparison、serial/resume、CPU/CUDA endpoint和runtime factory。事实背景采用主线程已读trace：PID12264在offset_shell/exchange接受4次更新后，`clean` 的 `isinstance(x,np.ndarray)` 报 `AttributeError: 'builtin_function_or_method' object has no attribute 'ndarray'`。本次**未SSH、未执行研究源码、未做数值/校验/benchmark、未修复或修改代码**；只新增此说明。不能据静态阅读断言故障只影响日志或数值未受影响。

## 五条发现

1. **报错指向np绑定异常，不是普通输入类型不受clean支持。** helpers第13行只有 `import numpy as np`；`clean:69–75` 没有np参数、局部赋值或global语句。若运行字节码与此源码一致，`np.ndarray` 的属性访问基对象当时是builtin function/method，才与该AttributeError一致。传入dict/list/ndarray的差别不能通过这段源码正常地把模块np变成函数。此信息仍不能说明是谁、何时改变了该绑定。

2. **在审查到的源码闭包中，没有找到合法的显式覆盖路径。** 未见 `original_exchange_helpers.np=...`、`clean.__globals__` 写入、对helpers字典的update、`exec/eval` 注入或 `global np` 重新赋值。helpers除NumPy外仅显式导入Path/contextmanager/ContextVar/dataclass/json/time/install/orth，没有wildcard导入。controller第46行 `from ... import clean` 取得函数，并不会把函数的global字典改成controller的字典。`run_comparison:13–14` 的 `exec_module(old)` 对象是独立的 `readonly_previous_shapes` 模块；`isolated_imports:49–56,60–82` 通过各自module spec导入helpers，并检查已加载来源路径；未见把旧runner字典合并进helpers的代码。该负面证据限于所读source，不覆盖未提供的启动器、运行时交互或已损坏的解释器状态。

3. **已声明runtime绑定与invocation状态不解释np被正常覆盖。** CUDA/serial入口显式修改的是 `r.new_model`、`r.RefinedCoreEndpointBatch` 和获批的 `r.CFG['wall_cap_s']`。helpers的 `invocation:37–45` 只set/reset `_INVOCATION` ContextVar；CONFIG读其candidate batch size，array_identity只增加该invocation的观察编号。它们均没有写np或clean的global字典。另一个module中同名np/local变量本身不会改helpers的np。当前serial入口每次仅一个job；不能仅因历史曾用双线程，就把本次错误归因为仍在发生的helpers线程共享写入。

4. **CUDA adapter未见明显的手写host-buffer生命周期违规，但这不是native安全证明。** `gpu_refined_endpoint_batch:30–33` 用 `detach().cpu().numpy()` 取得host数组；第86行raw residual另作copy，第87/104行只取小flags；第92/107行把active索引交给CUDA LongTensor。第114行packed结果回host后，第130行以NumPy赋值复制进返回steps/jsteps，audit第115–124行只保留lists/scalars。原EndpointBatch构造第45–48行从copy上传并同步；core第34/39行保留V并从copy上传。所读路径没有手写指针、ctypes缓冲区重映射、frombuffer、non_blocking传输、pinned-memory复用、自建CUDA stream或显式释放底层storage；按常规NumPy/PyTorch对象所有权语义，没有看到函数结束即悬空的host view。源码里的ctypes只用于Windows进程affinity/priority。静态阅读无法排除CPython/NumPy/PyTorch/native库错误、运行时内存破坏或不同部署版本，也没有证据把这些可能性定为根因。

5. **异常出现在日志转换，不足以界定影响仅为日志。** controller调用clean的位置包括receiver_cost第67行、round checkpoint第122行、online_result第135行，moves写入也经helpers第150行clean；这些位置与端点、Jd及内部接受过程交错。更关键的是同一helpers np还用于 `array_identity:58`、`basis:100`、anchor gain的 `score_from_outputs:102–105` 和Neighborhood的数组构造/有限性判断。如果helpers全局np确实异常，该共享命名空间并非只服务日志；也不能由一处Python异常判断故障前的数值对象是否保持完整。因此本说明既不认定已有接受prefix错误，也不为其数值完整性背书。主线程保留故障产物、废弃未提交tail、fresh同源进程从chi04接续并继续计费，是已声明恢复边界；fresh启动本身不是根因诊断或旧prefix验证。

## 仍不确定

本次没有故障瞬间的 `clean.__globals__['np']` 对象/写入历史、实际loaded module身份、native内存诊断或远端部署字节证据；按要求也没有新增检查来取得它们。当前只能记录：**源码未发现预期的np覆盖路径，运行时却报告了不符合该模块正常绑定的对象类型；原因与数值影响范围均未确定。** 无法在“未审查运行时写入/部署差异”和“解释器或native扩展状态异常”等可能性间作出证据支持的排序。此前另一次native access violation也不能单凭时间先后证明与本次同因。
