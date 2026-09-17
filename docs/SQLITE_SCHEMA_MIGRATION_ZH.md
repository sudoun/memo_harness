# v0.9.0：SQLite v3 Lineage-aware Durable Runtime

## 目标

v1 只持久化 observation。Calibration monitor 仍在内存中，因此进程重启会
丢失 prediction/outcome 状态，而且 `monitor.track()` 与 `store.append()` 之间
只能做补偿删除，无法抵抗进程崩溃。

v2 将 memory 与 monitor journal 放在同一个 SQLite 数据库；v3 在不重写
旧 posterior 的前提下补充证据血缘：

- `observations`：原始、带时间有效性的局部 posterior；
- `relation_laws`：数据库级 relation type → 完整有限关系律指纹锁；
- `monitor_runs`：规范化的冻结 monitor 配置及 SHA-256；
- `monitor_laws`：每个 run 对数据库级关系律的关联快照；
- `monitor_predictions`：posterior、标签顺序、关系律指纹的不可变快照；
- `monitor_outcomes`：首次真值与显式 correction 的不可变事件链。

`observations` 在 v3 新增：

- `source_event_id`：产生 observation 的直接事件；
- `lineage_root_id`：跨 parser、model echo 和 summary 共享的源事件根；
- `derived_from_json`：上游 observation ID 数组；
- `encoder_revision` 与 `calibration_revision`：posterior 入口版本。

新检索首先按 `lineage_root_id → source_event_id → evidence_family` 的顺序
确定 independence key，因此不同 family 对同一事件的复述只贡献一次
likelihood。旧行的新字段为 `NULL`/`[]`，继续使用原 `evidence_family` 语义。
写入时要求所有 `derived_from` parent 已存在于相同 namespace/relation type；
独立 echo 还必须保持相同 lineage root 与关系端点。parent-before-child 的追加
顺序使 lineage ancestry 无法形成环。

## 原子提交

`SQLiteMemoryStore.append_monitored()` 在一个 `BEGIN IMMEDIATE` 事务中写入
observation 和 prediction。任何一条 INSERT 失败都会回滚，因此重启后只可能
看到 `(0, 0)` 或 `(1, 1)`，不会看到半条监控记录。

Outcome 只执行追加。Correction 必须：

1. 显式给出非空原因；
2. supersede 当前 leaf；
3. 与原 observation、relation type、label order 和 law fingerprint 一致；
4. 时间不早于前驱事件。

partial unique root index、unique supersedes index 和验证 trigger 共同防止同一
observation 出现多个 root、分叉 correction 或跨 observation supersede。
`memory_schema_migrations` 以及上述五张 registry/journal 表的
UPDATE/DELETE 均由 trigger 拒绝。完整规范化 DDL 会在启动时核对；同名空
trigger、额外绑定到锁定表的 trigger、错误的索引/FK/CHECK 都会 fail closed。

## 配置与关系律锁定

同一个 `run_id` 首次打开时保存 canonical `MonitorConfig` 和 SHA-256。以后使用
不同阈值、窗口或容量打开会 fail closed。每条 prediction 同时保存完整 label
顺序及 finite-law fingerprint；注册同名但不同组合表的关系律也会被拒绝。
`relation_laws` 是数据库级约束，不能通过更换 `run_id` 绕过。首次绑定必须
提供完整、通过有限群律验证的 canonical law payload，不能只提交任意 SHA-256。

对已有 legacy observations 做首次关系律绑定时，会在同一个 immediate
transaction 内完整解码全部匹配行，并检查 posterior/prior support size。相同
support size 无法反推出历史 composition semantics，因此这一步仍是运维人员
依据原部署配置所做的 attestation。

## 迁移

当前 `PRAGMA user_version=3`。迁移全程在一个 immediate transaction 内完成：

1. version 0 无表：建立 v1 observation baseline；
2. version 0 legacy：验证必需列和 `observation_id` 主键后接管；
3. version 1：保留全部 observation，新增 v2 journal；
4. version 2：验证完整 monitor schema，追加 v3 lineage columns；
5. 写入严格 migration history `[1, 2, 3]`；
6. 验证必需表、列、索引、trigger 与终态版本后提交。

任一步失败都会 rollback。未来版本拒绝降级；partial v2 schema 不会被静默
当作成功迁移。

迁移事务提交后，运行时会切换并核验 WAL journal mode。由于另一进程可能恰好
在迁移提交与 WAL 切换之间获得 SQLite 锁，这一步仅对
`SQLITE_BUSY`/`SQLITE_LOCKED` 做有界指数退避，并且每次使用新连接；其他
I/O、权限或损坏错误不会被吞掉。跨进程测试覆盖了并发首次建库和读取锁释放后
的 WAL 重试。

## 运维命令

显式迁移：

```powershell
posterior-memory-schema --db memory.sqlite migrate
```

只读检查已有 schema（不创建、不迁移）：

```powershell
posterior-memory-schema --db memory.sqlite check
```

数据与 journal 核验：

```powershell
posterior-memory --db memory.sqlite verify
```

`verify` 运行 SQLite `quick_check`、`foreign_key_check`、schema compatibility、
prediction/outcome payload hash 和 correction-root 检查；它只报告问题，不修复。

## 已覆盖的回归

- 新数据库建立与重复打开；
- v0/v1 到 v2 无损迁移；
- future/malformed schema fail closed；
- 8 次并发首次打开；
- observation/prediction 中途失败的整事务回滚；
- 重启前后 monitor report 逐字段一致；
- hash-locked config 与 law mismatch 拒绝；
- 并发首次 outcome 只有一个 durable winner；
- correction 前后 `as_of` 快照及旧事件保留；
- append-only trigger 和 artifact verifier。
- 同名 no-op/额外 trigger、伪造 history 与未绑定 relation type 拒绝；
- 整数时间戳 canonical hash、跨 worker sequence-hole 与 stale cache 恢复；
- 真实 subprocess 并发迁移、事务 failpoint 回滚和 CLI 重启。
- v2 到 v3 的 lineage column 无损迁移；
- cross-family lineage echo 去重、SQLite round-trip 与 artifact verify；
- `recent/coverage/cycle_aware` 在 InMemory/SQLite 后端选择一致。
