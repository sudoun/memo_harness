# v0.8.0：Durable Calibration / Drift Monitor

## 设计目标

局部 posterior 是 Posterior Memory Harness 的信息入口。若 encoder 在新措辞、
新工具版本或新路径族上变得过度自信，全局组合律无法恢复已经被压到接近零的
真实类别概率。

本模块监控该入口，但明确不做三件事：

- 不根据线上窗口自动重拟合 temperature；
- 不修改已存 observation；
- 不把监控报告自动注入模型 prompt。

因此 monitoring 不会悄悄变成一个使用测试反馈做模型选择的通道。

## API

```python
from posterior_memory_harness import (
    CalibrationDriftMonitor,
    MonitorConfig,
    PosteriorMemoryHarness,
    SQLiteMemoryStore,
    cyclic_law,
)

monitor = CalibrationDriftMonitor(MonitorConfig(
    reference_size=200,
    current_size=100,
    min_current_size=30,
    run_id="production-v1",
))
harness = PosteriorMemoryHarness(
    SQLiteMemoryStore("memory.sqlite"),
    monitor=monitor,
).register_relation("task-phase", cyclic_law(4))

# observe(...) 后，在工具验证、人工反馈或环境结局到达时回填：
harness.record_outcome("observation-id", "true-relation-label")
report = harness.calibration_report(namespace="agent-run")

# 真值修正只能追加，不允许覆盖：
harness.correct_outcome(
    "observation-id",
    "corrected-label",
    reason="human adjudication",
)
```

只有 `independent_evidence=True` 的原始 observation 会进入监控；compaction
等派生摘要不会被当成第二个预测样本。

## 固定 reference 与 current window

每个 relation type 独立计算，不能跨不同 probability support 聚合：

- 最早的 `reference_size` 个已标注预测构成冻结 reference；
- 其后的最新 `current_size` 个预测构成 current；
- current 少于 `min_current_size` 时状态为 `insufficient_data`，不会伪报
  `healthy`。

每个窗口报告：

- accuracy；
- NLL；
- multiclass Brier score；
- equal-width ECE；
- top-2 coverage；
- mean confidence；
- mean entropy。

预设 guard 检查：

- NLL 相对 reference 的增量；
- Brier 相对 reference 的增量；
- current ECE 的绝对上限；
- reference/current mean posterior marginal 的 Jensen--Shannon divergence。

报告同时按 `relation_type` 和 `source_family` 分层。`source_family` 依次取自
provenance 的 `monitor_family`、`encoder_family` 或 `tool` 字段。

## Locked audit

第一轮 seeds 1100--1109 使用 `prediction_js=0.03`：

- stable false-alert：1/10；
- overconfidence detection：10/10；
- prediction-shift detection：5/10，**失败**。

该结果保留在 `benchmark/monitoring_audit/development_results_v1/`，没有覆盖。
它被降为 development pilot。

随后固定新阈值 0.01，并仅使用全新 seeds 2100--2109 做一次 fresh audit：

| 条件 | 告警率 | 主要平均变化 |
|---|---:|---:|
| Stable | 0/10 | posterior JS 0.00020 |
| Overconfident | 10/10 | NLL +0.18785，ECE 0.16257 |
| Prediction shift | 10/10 | posterior JS 0.02754 |

fresh audit 的三个预设 guard 全部通过，文件位于
`benchmark/monitoring_audit/fresh_locked_results_v2/`。

这只是 monitor mechanism audit，不是“LLM Agent 记忆能力已经提升”的现实任务
证据。真实收益仍需固定模型、固定任务、固定 tool harness 的长期成对实验。

## v0.8 Durable 语义

- SQLite 模式会原子写入 observation 与 prediction snapshot；
- `MonitorConfig` 按 `run_id` hash-lock，重启改变 guard 会被拒绝；
- relation law 在数据库级锁定，切换 `run_id` 不能绕过；
- prediction 固化 posterior、label order 与 relation-law fingerprint；
- outcome 和 correction 是 append-only event chain，数据库禁止 UPDATE/DELETE；
- `as_of` 同时按 prediction/outcome 的 knowledge time 截断，未来 correction
  不会泄漏进旧报告；
- 重启从 journal 确定性重放，报告包含 `config_sha256` 和 outcome high-water
  sequence，以及 prediction high-water sequence。

纯 `InMemoryMemoryStore` 仍保持轻量、进程内语义；需要长期运行或多进程 CLI
时必须使用 SQLite。

## 当前边界

- 没有 ground truth 时只能监控 posterior marginal/confidence，不能计算
  calibration proper scores；
- prediction-marginal drift 也可能来自真实 label-prior shift，告警表示需要
  诊断，不等于 encoder 必然损坏；
- 阈值必须在 audit 前固定，不能根据 locked 测试结果反复调整；
- 当前 outcome-before-observation 会原子拒绝并要求调用方重试，尚未提供
  pending-outcome queue；
- SQLite journal 与 memory store 必须位于同一数据库；跨服务 telemetry 需要
  后续 outbox/stream adapter。
