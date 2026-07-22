# W90Flow 本地长时运行监督器

这一层解决的是“提交后怎样等待、隔多久检查、怎样自动进入下一阶段、中断后怎样继续”。它是确定性 Python 代码，不需要 AI 在运行中做决策。

## 运行模型

```text
supervisor lock
  └─ SCF child ──轮询 PID/输出──> SCF PASS 边界
       └─ Band child ─轮询 PID/输出──> Band PASS 边界
            └─ Candidates → Wannier finite search → Quality → Rank → Export
```

每一个 child 只负责推进到一个指定阶段边界。child 退出且该阶段 `stage.json=PASS` 后，supervisor 才可以提交下一阶段。因此墙钟截止或用户请求暂停时，不需要破坏 CHGCAR/WAVECAR/MMN 等未写完文件。

## 命令

先预览，不写状态、不启动引擎：

```bash
运行W90Flow.command supervise RUN_DIR --until export
```

实际运行一个最长 12 小时的本地作业：

```bash
运行W90Flow.command supervise RUN_DIR --execute \
  --until export \
  --poll-seconds 30 \
  --heartbeat-seconds 300 \
  --no-progress-seconds 1800 \
  --stage-timeout-seconds 86400 \
  --max-hours 12
```

只读查看：

```bash
运行W90Flow.command supervisor-status RUN_DIR
```

## 状态和日志

- `supervisor-state.json`：监督器/child/worker PID、当前阶段、进展年龄、告警、checkpoint 和下一动作。
- `supervisor.log`：阶段提交、定期心跳、完成边界和暂停原因。
- `supervisor-child-STAGE.log`：每个阶段的 W90Flow 子进程输出。
- `.w90flow-supervisor.lock`：防止同一 run 被两个长时监督器重复提交。
- `.w90flow.lock`：由实际状态机child持有，防止外部命令重复启动同一 run。

## 检查逻辑

`poll-seconds` 每次检查：

1. child PID 是否仍存在；
2. `stage.json/attempt.json` 中的实际 VASP/Wannier worker PID；
3. `runner.log`/`OUTCAR`/`OSZICAR`/`vasprun.xml`/`wannier90.wout` 等文件的大小和更新时间；
4. 该阶段最终是否 `PASS`。

`heartbeat-seconds` 只控制多久把完整快照写入状态/日志，不影响引擎。

## 异常与安全边界

- `NO_PROGRESS_TIMEOUT`：发现指定时间内没有监测到输出更新。
- `STAGE_TIMEOUT`：单个阶段的墙钟时间超过限额。
- `SIGINT/SIGTERM`：记录用户/系统请求暂停。
- `max-hours`：达到总墙钟截止。

上述事件均只设置“当前阶段完成后停止衔接”；不会自动改投影、能窗、k 网格、INCAR 或质量门槛。超时/无进展后状态为 `BLOCKED_ALERT_BOUNDARY`，用户信号或墙钟截止后为 `PAUSED_BOUNDARY`。

## 断点恢复

同一条 `supervise ... --execute` 命令即是恢复入口。

- 已经 `PASS` 的阶段直接跳过；
- 旧 supervisor 消失但 child PID 仍在时，新 supervisor 只附着等待，不重复提交；
- 旧 child 结束后，只有完整阶段已 `PASS` 才继续；否则安全 `BLOCKED`；
- VASP/Wannier worker 的更细粒度完整输出回收和一次基础设施恢复，由 `run/resume` 状态机执行。

## 关机、合盖和拔盘

supervisor 只能防止重复并保存软件断点，不能在 Mac 睡眠或移动硬盘断开后继续计算。需要拔盘时，先发出一次中断并等待 `PAUSED_BOUNDARY/BLOCKED_*` 且 `supervisor-status` 中 supervisor、child、worker 全部不存活，再弹出移动硬盘。

