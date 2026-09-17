# PosteriorGlue Agent Memory Harness：受控验证报告

## 结论

在固定局部观测、固定查询图和固定 Agent 输出规则下，保留完整 posterior
并使用正确非交换关系律，显著提高了结构化 Agent 状态记忆能力。

本轮是 **memory-layer mechanism benchmark**：局部 observation posterior
由受控生成器产生，用于隔离记忆推断机制；它不是开放域 LLM benchmark，
也没有声称语言前端已经解决。

## 协议

- 关系后端：非交换群 \(S_3\)，表示 Agent 五个 checkpoint 间的有序状态变换；
- 20 seeds，每个 seed 200 episodes；
- cycle-free：4 条 chain observations；
- cycle-rich：相同 chain 加 4 条冗余约束；
- 每条 observation 保存完整 6 类 posterior、uniform local prior、
  provenance 和 evidence-family；
- 查询时执行 prior correction，并保证每个 evidence-family 只进入一次；
- 全部方法复用相同的局部 posterior。

局部 relation top-1 accuracy 为 **64.76%**，true relation top-2 coverage
为 **99.98%**。因此该设置模拟了“语言/观测前端通常保留真实候选，但局部
argmax 经常错误”的 Agent memory 场景。

## 主要结果

### Cycle-rich memory graph

| 方法 | Node accuracy | Endpoint accuracy | Entire-episode success | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| Hard top-1 memory | 65.59% | 68.95% | 34.28% | 4.076 | 0.564 |
| Full posterior + correct law | **94.45%** | **94.95%** | **88.33%** | **0.195** | **0.093** |
| Full posterior + wrong order | 66.53% | 75.18% | 24.95% | 0.952 | 0.470 |
| Stratified shuffled posterior | 17.13% | 17.68% | 0.05% | 2.981 | 1.174 |
| Oracle local relation | 100.00% | 100.00% | 100.00% | 0.000 | 0.000 |

Full posterior 相对 hard top-1 的 paired difference：

- Node accuracy：**+28.86 pp**，95% CI **[+27.89,+29.84]**；
- Endpoint accuracy：**+26.00 pp**，95% CI **[+24.72,+27.28]**；
- Entire-episode success：**+54.05 pp**，95% CI **[+52.65,+55.45]**；
- NLL：**-3.880**，95% CI **[-4.045,-3.716]**；
- Brier：**-0.471**，95% CI **[-0.486,-0.456]**。

### Cycle-free negative control

没有冗余 cycle 时：

- hard top-1 node accuracy：39.44%；
- full posterior node accuracy：40.38%；
- paired gain：**+0.94 pp**，95% CI **[+0.50,+1.35]**；
- 但 NLL 从 8.780 降到 1.309，Brier 从 1.211 降到 0.656。

这复现了论文中的结构规律：没有冗余约束时，完整 posterior 主要改善 proper
scores；当多个 observation 共同约束同一状态时，次优概率质量才会转化为
显著的决策准确率收益。

## 机制控制

- Correct law − wrong order：node accuracy **+27.93 pp**，
  95% CI **[+27.04,+28.68]**；
- Correct sample correspondence − stratified shuffle：
  node accuracy **+77.32 pp**，95% CI **[+76.66,+77.99]**；
- Oracle − full posterior：node accuracy **+5.55 pp**，
  说明当前主要剩余瓶颈仍是局部 observation posterior。

因此，增益不能仅用“图里 observation 更多”解释：同一批 posterior 使用错误
顺序律会明显失败；保留 sharpness 但打乱 observation 与 episode 对应关系也会
降至接近六分类 chance。

## 对 LLM Agent 的准确含义

本实验支持以下结论：

> 当 Agent 的长期状态可以表示为有序关系图，局部记忆保留真实候选概率，并且
> 存在冗余一致性约束时，PosteriorGlue memory harness 能显著优于只保存局部
> top-1 事实的记忆系统。

本实验尚不支持：

- 开放域对话记忆一定提升 28.86 pp；
- 任意自然语言事实都具有可用的关系律；
- LLM 自报 confidence 可以直接作为 calibrated posterior；
- memory retrieval、实体解析和 prompt injection 已经解决。

下一步必须冻结本轮 inference core，只替换 `ObservationEncoder`：

1. 使用真实 LLM/小型 relation classifier 产生 top-k relation posterior；
2. 在 development-only 数据上校准；
3. 比较 vector RAG、summary memory、hard graph memory 与本模组；
4. 在新的 locked episode audit 上只运行一次。

## 文件

- `memory_harness.py`：模型无关 memory core；
- `test_memory_harness.py`：关系律、组合、证据去重和安全 capsule 测试；
- `run_controlled_benchmark.py`：20-seed 受控 benchmark；
- `seed_metrics.csv`：seed-level 指标；
- `absolute_bootstrap.csv`：绝对指标及 95% CI；
- `paired_bootstrap.csv`：paired difference 及 95% CI；
- `diagnostics.csv`：局部 top-1/top-2 诊断；
- `config.json`：正式运行配置。
