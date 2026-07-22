# Wannier Workflow（W90Flow）

这是从优化后 POSCAR 开始的确定性、AI-free 核心。结构集合可以有 1、2 或更多成员；`independent` 分别排名，`common_basis` 才强制共同投影，程序不假定材料是铁电体或一定存在 L1/L2。

> 当前状态：**private prepublication development**。不得在未完成 [DISCLOSURE.md](DISCLOSURE.md) 审查时改为 public。测试数据分层见 [FIXTURE_POLICY.md](FIXTURE_POLICY.md)。

## 直接使用

从一个干净 clone 安装并运行不含私有数据的单元测试：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/audit_repository.py
```

仓库 clone 后默认只使用 `tests/fixtures/` 中的合成夹具。本地私有的真实回归集用环境变量挂载，不会被复制进仓库：

```bash
W90FLOW_REFERENCE_ROOT=/path/to/reference-fixtures \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

常用命令：

```bash
w90flow plan /path/to/project.yaml
w90flow audit-reference /path/to/reference-case
w90flow window-suggest /path/to/EIGENVAL --num-wann 9 --target-min -4.6 --target-max -0.5
w90flow candidate-suggest examples/BiTeI-orbital-scores.example.json --soc
w90flow pyw90-baseline /path/to/EIGENVAL --num-wann 9 --target-min -4.6 --target-max -0.5 --fermi 0 --energy-reference absolute
```

创建一个不可覆盖的状态目录：

```bash
PYTHONPATH=src python3 -m w90flow init ../03_自动化架构/configs/pair-example.yaml --runs-root runs
```

每个 run 固定写入配置快照、POSCAR/POTCAR-lock 哈希，以及八个阶段的 `stage.json`。macOS/ExFAT 的 `._*` 不会成为科学输入。

## 实际执行（v0.6）

`prepare` 校验私有 POTCAR component/combined SHA256，建立不可覆盖 run，但不启动计算：

```bash
w90flow prepare /path/to/project.yaml --runs-root /path/to/private/runs
```

完成 `prepare` 后，一条命令可以从当前断点确定性推进 SCF 到 top-K 导出：

```bash
运行W90Flow.command run RUN_DIR
运行W90Flow.command run RUN_DIR --execute
运行W90Flow.command resume RUN_DIR --execute
运行W90Flow.command inspect RUN_DIR
```

默认是 dry-run；只有加 `--execute` 且配置中 `execution.allow_execute: true` 才运行。`run` 与 `resume` 共用同一状态机，会拒绝同一 run 的重复监督器，并对基础设施中断只允许一次同输入恢复。

长时运行请用 Python 监督层，它会在每个完整阶段边界自动衔接，中间按设定间隔轮询 PID、输出更新和阶段状态：

```bash
运行W90Flow.command supervise RUN_DIR
运行W90Flow.command supervise RUN_DIR --execute \
  --poll-seconds 30 --heartbeat-seconds 300 \
  --no-progress-seconds 1800 --stage-timeout-seconds 86400 \
  --max-hours 12
运行W90Flow.command supervisor-status RUN_DIR
```

`supervise` 也默认 dry-run。第一个 `SIGINT/SIGTERM`、墙钟截止、无输出进展或阶段超时都不会自动修改投影、能窗、INCAR 或质量门槛；正在运行的引擎先到达完整阶段边界，然后写入 checkpoint 并停止衔接下一阶段。详细状态文件和恢复规则见 [SUPERVISOR.md](SUPERVISOR.md)。

top-K 交付后的 shift-current/SHG 作业规划也已有默认 dry-run 的本地 Python 层：

```bash
PYTHONPATH=src python3 -m w90flow.response_plan build response-campaign.json
PYTHONPATH=src python3 -m w90flow.response_plan preview RESPONSE_PLAN_DIR
PYTHONPATH=src python3 -m w90flow.response_plan run RESPONSE_PLAN_DIR
# 只有复核所有响应参数后才加 --execute
```

它硬性使用 `NOCC=nocc_model`，按响应类型→rank→结构串行，记录 MATLAB PID/日志和可恢复状态；不会将 `nocc_target` 误用为完整 hr 模型的占据数。详见 [RESPONSE_AUTOMATION.md](RESPONSE_AUTOMATION.md)。

Wannier 拟合与非线性光学求和器保持为两个独立程序。核心交付的最小文件接口见 [NLO_HANDOFF.md](NLO_HANDOFF.md)；当前 Python shift-current/SHG 以后只需按该接口读取 top-K 包，不属于 W90Flow v1.0 的通过条件。

下面的分步命令仍保留，用于调试和审计：

```bash
运行W90Flow.command run-stage RUN_DIR scf
运行W90Flow.command run-stage RUN_DIR scf --execute
运行W90Flow.command run-stage RUN_DIR band --execute
运行W90Flow.command generate-candidates RUN_DIR
运行W90Flow.command run-stage RUN_DIR wannier --execute
运行W90Flow.command evaluate RUN_DIR
运行W90Flow.command finalize RUN_DIR
```

候选的有限 exact/disentanglement 尝试、统一目标区判分和三层排名：

```bash
运行W90Flow.command plan-attempts RUN_DIR
运行W90Flow.command run-attempt RUN_DIR STRUCTURE CANDIDATE attempt-001
运行W90Flow.command run-attempt RUN_DIR STRUCTURE CANDIDATE attempt-001 --execute
运行W90Flow.command evaluate-attempt RUN_DIR STRUCTURE CANDIDATE attempt-001
运行W90Flow.command rank-attempts RUN_DIR
运行W90Flow.command export-ranked-attempts RUN_DIR
```

`rank-attempts` 分开记录 PROCAR/PyW90 试算优先级、真实拟合排名和光学完备基组建议。`export-ranked-attempts` 只导出 POSCAR、hr、纯 Wannier 中心、比较图、win 和质量/哈希清单，不含 POTCAR/CHGCAR/MMN。最简流程图见 [WORKFLOW_ONE_PAGE.md](../01_流程规范/WORKFLOW_ONE_PAGE.md)。

`pyw90-baseline` 只将 PyW90 `eig suggest` 的终端表转成带工具/输入哈希的 JSON 对照。它不覆盖 W90Flow 内部状态数约束，也不能放宽质量硬门槛。

质量文件同时写 `nocc_target` 和 `nocc_model`：前者只用于公共能区拟合判分，后者才是完整 hr 模型交给 shift-current/SHG 代码的 `NOCC`。二者不得混用。

`reconcile RUN_DIR scf|band|wannier` 只根据完整输出回收后台状态，不会重复提交。

已有 Wannier attempt 只有在 `POSCAR + POTCAR + INCAR + KPOINTS + CHGCAR` 五个 SHA256 全部相同时才能复用；复用后仍按新项目的目标能区和门槛重新评价，不导入旧排名：

```bash
运行W90Flow.command import-attempt RUN_DIR STRUCTURE CANDIDATE attempt-001 \
  --source /absolute/completed/attempt
```

只要投影、`NUM_WANN`、排除带、能窗或收敛参数造成 INCAR 变化，命令就拒绝复用并要求真实拟合。

当 SCF/Band 的 `POSCAR + POTCAR + INCAR + KPOINTS` 四个哈希完全一致时，可以零复制复用已完成阶段；任何一个输入不同都会拒绝：

```bash
运行W90Flow.command import-stage RUN_DIR scf --source L1=/absolute/old/L1/scf --source L2=/absolute/old/L2/scf
运行W90Flow.command import-stage RUN_DIR band --source L1=/absolute/old/L1/band --source L2=/absolute/old/L2/band
```

C2DB 材料池筛选始终以 SQLite 只读模式打开源数据库：

```bash
w90flow material-pool /path/to/c2db.db /path/to/material-pool.yaml \
  --output-dir /path/to/output-pool
```

v1.0 的冻结范围、输入/输出、有限重试表和验收门槛见 [AUTOMATION_V1_SPEC.md](AUTOMATION_V1_SPEC.md)。

## 已完成与下一步

已完成：

- 配置和 POSCAR/POTCAR-lock 校验；
- SCF→Band→候选→Wannier→质量→排名→交付阶段契约；
- EIGENVAL、OUTCAR、hr、centres、wout 关键解析；
- 外窗每个 k 点至少 `NUM_WANN` 状态、冻结窗不超过 `NUM_WANN` 的硬约束；
- 同一候选必须覆盖所有相关结构，按最差一侧门槛/误差排序；
- 由目标能区积分轨道权重生成有限的覆盖率/化学完备候选分支，并按 SOC 自动计算 `NUM_WANN`；
- 无人工价轨道答案时也自动保留通用 p-only、complete-sp 和 all-relevant 分支；重复投影会自动去重；
- 三个现有实例作为回归测试。
- 私有 POTCAR 拼接/哈希验证、SCF→Band CHGCAR 硬链接交接和显式执行保护；
- 从真实 EIGENVAL+PROCAR 自动积分轨道权重；
- exact-band 模型的通用 DFT/Wannier 误差、带隙、HR/中心、spread 检查与 PDF；
- 单结构/共同基底的最小排名和无 POTCAR export。
- 用 SCF 均匀网格与高对称路径共同判断 exact band block；
- 不可覆盖的候选 attempt、真实执行、失败分类与定向下一动作；
- 所有候选在同一目标能区判分，输出试算优先级/拟合排名/光学基组建议三张逻辑榜。
- `common_basis` 从所有结构的 SCF/Band 能量和平均轨道特征共同规划同一 attempt/能窗；联合排名禁止混用不同重试参数，并按最差成员误差排序。
- exact 判定的最小全局边界间隔可配置；InSe 以 0.05 eV 门槛强制进入真实解纠缠分支。
- 已完成阶段可在四个科学输入哈希完全一致时用硬链接导入，避免重复 SCF/Band 和大型文件复制。
- MPI/沙箱等基础设施秒退与数值失败分开分类；仅前者允许同一 attempt 一次审计化重试。
- C2DB 只读筛选保留来源 ID/路径、筛选字段和复核标记，不使用会重排 POSCAR 元素或删除旧目录的旧脚本逻辑。
- 支持 `energy_reference: fermi_relative`：候选权重和能窗在统一的 `E-E_F` 坐标中规划，每个结构在写入 Wannier90 前自动转成绝对能量，并同时保存参考计划和物化后计划。
- 新增单命令 `run/resume`、进程级 PID 断点、单 run 锁、有限分支选择和机器可读 `workflow-summary.json`。
- 新增纯 Python `supervise`：阶段边界调度、可配置轮询/心跳、PID 恢复、无进展/超时告警、墙钟截止和信号安全 checkpoint。
- 新增 top-K→MATLAB 响应计划层：自动传递 `nocc_model`、串行作业、PID/指纹断点和旧 MATLAB 活进程附着等待。

数据集与跨数据库 benchmark 的字段、切分和 ML 路线见 [DATASET_AND_BENCHMARK.md](DATASET_AND_BENCHMARK.md)。

为了防止根据三个已知答案不断补特例，候选规则的开发集/冻结盲测集/外部盲测集和评分方式见 [GENERALIZATION_PROTOCOL.md](GENERALIZATION_PROTOCOL.md)。

下一步只收敛 v1.0：端到端模拟/中断测试、PyW90 对照 adapter、BiTeI/InSe 输出回放和一次小型冷启动。生产网格与 top-K 下游 shift-current/SHG 在 v1.0 验收后再展开。

## 科学边界

能窗“合法”只表示状态数满足数学约束，不表示 Wannier 拟合已经优良。最终候选仍必须通过插值误差、带隙/连接性、spread/中心、Hermiticity、双结构共同基底，以及 shift current/SHG 稳定性检查。
