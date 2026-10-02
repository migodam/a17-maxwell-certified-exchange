# Tiny proposer preparation

实现合同：IMPLEMENTATION_CONTRACT.md；冻结配置：FROZEN_CONFIG.json；源码：tiny_proposer.py。当前 PREPARED_TOY_ONLY，真实读取/训练未运行。

TOY_CHECKS.json 中11项实施检查通过：禁止字段不进features、未授权前不读manifest/test、validation不能拟合normalization、两模型FP64/FP32候选排列等变、六runs loss与heldout priorities精确可重复、全部runs保留、test label不影响train normalization、cost speedup保持null。toy_six_runs_frozen/ 保存两个模型×三个seed的100epoch toy state_dict、loss、priority/metrics和费用receipt；这些全是人工toy，不是Maxwell、NN泛化或科学结果。

真实入口只在父线程明确四对象NN gate PASS后使用；OWNER_AUTH与DATASET_MANIFEST模板当前未填且阻断。不要把原两对象 expansion gate替代授权。运行需现有 Gaussian/.venv_nn/bin/python 的CPU PyTorch，无安装/新框架。

复现toy检查使用 `check_toy.py --out <新的toy子目录>`，保护已有输出；早期 toy_six_runs、v2、final 保留为实施版本记录，最终冻结检查只绑定 toy_six_runs_frozen。真实输出也要求本目录内新目录。源码/配置hash、runtime版本、模型hash保留于receipts；输入真实manifest行数/hash及feature取得费用须由父线程绑定。

同 checkpoint 的完整候选 Qhat top2 baseline 单独记录为 QHAT_TOP2_BASELINE.json；每 run 同时保存按对象内机会求和的 capture/regret，保持no-op和零机会missing原因。
