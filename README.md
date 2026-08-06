# Wannier Workflow（W90Flow，Stage 1）

这是从优化后 POSCAR 开始的确定性、AI-free 核心。结构集合可以有 1、2 或更多成员；`independent` 分别排名，`common_basis` 才强制共同投影，程序不假定材料是铁电体或一定存在 L1/L2。

本仓库只负责 SCF/Band、候选、真实 Wannier 拟合、质量和 top-K 交付。新的 shift-current/SHG Python 核在相邻的 `../09_非线性光学Python/`，两者只通过 [NLO_HANDOFF.md](NLO_HANDOFF.md) 的冻结 manifest 接口连接。`response_plan.py`/`matlab_backend.py` 是迁移期兼容旧 MATLAB 队列的桥，暂不作为 Stage 2 新功能入口。

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

### 项目验收主入口：优化好的 POSCAR → Stage-1 Top-K

`stage1-coldstart` 是当前项目的核心入口。一个 YAML 同时冻结 SCF/Band 前处理和
Wannier 有限候选合同；程序会保存唯一 preprocess run 和唯一 Wannier run 指针，
中断后只恢复同一个目录，不重复创建作业。

```bash
PYTHONPATH=src python3 -m w90flow stage1-coldstart /path/to/stage1.yaml \
  --campaign-dir /path/to/campaign-state

# 复核 dry-run 后才执行；重复这条命令就是断点续跑
PYTHONPATH=src python3 -m w90flow stage1-coldstart /path/to/stage1.yaml \
  --campaign-dir /path/to/campaign-state \
  --runs-root /path/to/runs \
  --execute
```

签名状态锁定 YAML、POSCAR、SCF/Band/Wannier 模板和 POTCAR lock 的 SHA256。
完成 Band 后，所有有限的投影候选都会进入真实 Wannier 门槛；某个投影在第一个
合格能窗后停止它自身的冗余窗口尝试，其他投影仍继续验收。Stage 2 不在该命令中。

默认 dry-run 也是完整、只读的提交前检查：它不会创建 campaign/run 或复制 POTCAR，
但会读取真实 POTCAR 组件并核对 POSCAR 元素顺序、组件 SHA256、拼接 SHA256 及命名
lock set；同时统一核对 `physics.soc`、SCF/Band 的 `preprocess.soc`、`backend.vasp`、
各 launcher 参数中的 `std/ncl` 和 MPI 数。启用 SOC 的 SCF/Band（以及启用时的 Relax）
模板必须显式提供恰好 `3*NIONS` 个 `MAGMOM` 分量。任一项不一致时，`--execute` 也会在
启动 VASP 和创建 campaign 状态前失败。
当前 launcher 合同要求命令参数列表的最后两项依次是 `std|ncl` 和正整数 MPI 数。

### 第一优先入口：从已有 SCF/Band 开始自动拟合

现阶段最稳定的入口是不重复电子结构，而是导入输入哈希一致的 SCF/Band 目录后自动完成候选、有限 attempts、质量、排名和 top-K：

```bash
w90flow fit-from-bands /path/to/project.yaml --runs-root /path/to/private/runs \
  --scf-source L1=/absolute/scf/L1 --band-source L1=/absolute/band/L1 \
  --execute
```

多结构时为每个 structure id 各给一组 `--scf-source/--band-source`。Band 除 POSCAR、POTCAR、INCAR、KPOINTS 外还必须和 SCF 的 CHGCAR 哈希一致。默认 dry-run；执行后直接推进到无 POTCAR 的 top-K package。

`prepare` 校验私有 POTCAR component/combined SHA256，建立不可覆盖 run，但不启动计算：

```bash
w90flow prepare /path/to/project.yaml --runs-root /path/to/private/runs
```

完成 `prepare` 后，一条命令可以从当前断点确定性推进 SCF 到 top-K 导出：

```bash
w90flow run RUN_DIR
w90flow run RUN_DIR --execute
w90flow resume RUN_DIR --execute
w90flow inspect RUN_DIR
```

默认是 dry-run；只有加 `--execute` 且配置中 `execution.allow_execute: true` 才运行。`run` 与 `resume` 共用同一状态机，会拒绝同一 run 的重复监督器，并对基础设施中断只允许一次同输入恢复。

长时运行请用 Python 监督层，它会在每个完整阶段边界自动衔接，中间按设定间隔轮询 PID、输出更新和阶段状态：

```bash
w90flow supervise RUN_DIR
w90flow supervise RUN_DIR --execute \
  --poll-seconds 30 --heartbeat-seconds 300 \
  --no-progress-seconds 1800 --stage-timeout-seconds 86400 \
  --max-hours 12
w90flow supervisor-status RUN_DIR
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

### 候选搜索：最终排名只有一种

每个项目都应使用**完整搜索**作为最终结果：PyW90/PROCAR 只决定候选先后，所有
预先声明的有限投影与能窗仍会完成，再按真实拟合质量导出 Top-K。所谓“优先搜索”
只是在完整搜索过程中先运行高优先级候选，使长作业较早产生临时 Top-K；它绝不改变
候选集合、质量门槛或最终排名。

SCF/Band 固定为单个重 VASP 阶段。它们完成后，不同 Wannier 候选目录彼此独立，可
按机器资源并行。先在目标队列做一次 `1×32`、`2×16`、`4×8` 的基准，再明确启用
并行；不要在 8 GiB 本地 Mac 上同时运行多个 VASP。

先查看环境，不启动计算：

```bash
w90flow resource-plan --candidates 7 --mpi-per-candidate 8 \
  --policy candidate_parallel
```

YAML 默认保持串行。通过队列基准后，才加入：

```yaml
execution:
  candidate_search:
    policy: candidate_parallel
    mpi_processes_per_candidate: 8
    max_workers: auto
```

`auto` 读取 Slurm 分配、CPU 亲和性和内存；32 核且内存充足时会得到 `4×8`，8 GiB
本地机器自动限制为一个 worker。并行只改变候选的运行顺序和资源分配，不改变 SOC、
投影、能窗或质量门槛。SCF/Band 以及输入哈希一致的 completed stage 仍按原有严格
规则复用；不同投影的 Wannier 输出只有 `POSCAR+POTCAR+INCAR+KPOINTS+CHGCAR`
全部一致才可复用。

下面的分步命令仍保留，用于调试和审计：

```bash
w90flow run-stage RUN_DIR scf
w90flow run-stage RUN_DIR scf --execute
w90flow run-stage RUN_DIR band --execute
w90flow generate-candidates RUN_DIR
w90flow run-stage RUN_DIR wannier --execute
w90flow evaluate RUN_DIR
w90flow finalize RUN_DIR
```

候选的有限 exact/disentanglement 尝试、统一目标区判分和三层排名：

```bash
w90flow plan-attempts RUN_DIR
w90flow run-attempt RUN_DIR STRUCTURE CANDIDATE attempt-001
w90flow run-attempt RUN_DIR STRUCTURE CANDIDATE attempt-001 --execute
w90flow evaluate-attempt RUN_DIR STRUCTURE CANDIDATE attempt-001
w90flow rank-attempts RUN_DIR
w90flow export-ranked-attempts RUN_DIR
```

`rank-attempts` 分开记录 PROCAR/PyW90 试算优先级、真实拟合排名和光学完备基组建议。`export-ranked-attempts` 只导出 POSCAR、hr、纯 Wannier 中心、比较图、win 和质量/哈希清单，不含 POTCAR/CHGCAR/MMN。冻结流程与验收边界见 [AUTOMATION_V1_SPEC.md](AUTOMATION_V1_SPEC.md)。

`pyw90-baseline` 只将 PyW90 `eig suggest` 的终端表转成带工具/输入哈希的 JSON 对照。它不覆盖 W90Flow 内部状态数约束，也不能放宽质量硬门槛。

质量文件同时写 `nocc_target` 和 `nocc_model`：前者只用于公共能区拟合判分，后者才是完整 hr 模型交给 shift-current/SHG 代码的 `NOCC`。二者不得混用。

`reconcile RUN_DIR scf|band|wannier` 只根据完整输出回收后台状态，不会重复提交。

已有 Wannier attempt 只有在 `POSCAR + POTCAR + INCAR + KPOINTS + CHGCAR` 五个 SHA256 全部相同时才能复用；复用后仍按新项目的目标能区和门槛重新评价，不导入旧排名：

```bash
w90flow import-attempt RUN_DIR STRUCTURE CANDIDATE attempt-001 \
  --source /absolute/completed/attempt
```

只要投影、`NUM_WANN`、排除带、能窗或收敛参数造成 INCAR 变化，命令就拒绝复用并要求真实拟合。

SCF 的 `POSCAR + POTCAR + INCAR + KPOINTS` 四个哈希完全一致时可以零复制复用；Band 还必须核验其 `CHGCAR`。任何一个输入不同都会拒绝：

```bash
w90flow import-stage RUN_DIR scf --source L1=/absolute/old/L1/scf --source L2=/absolute/old/L2/scf
w90flow import-stage RUN_DIR band --source L1=/absolute/old/L1/band --source L2=/absolute/old/L2/band
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
- 受保护目标能区内的 DFT 能带数不得超过 `NUM_WANN`：提交前直接淘汰不可能完备的候选，拟合后再次用实际 Wannier 模型维数复核，并以 `INCOMPLETE_TARGET_SUBSPACE` 留下可审计失败记录；
- 同一候选必须覆盖所有相关结构，按最差一侧门槛/误差排序；
- 由目标能区积分轨道权重生成有限的覆盖率/化学完备候选分支，并按 SOC 自动计算 `NUM_WANN`；
- 可选 `candidates.pyw90_dos.enabled: true` 将 PyW90 `pre dos` 投影作为带工具/输入哈希的第二候选来源；重复投影去重，所有建议仍必须经过相同真实拟合与硬质量门槛；
- 无人工价轨道答案时，从 PROCAR 自动生成 s/p/d/f 单壳层和完备壳层组合；p-only/complete-sp 只是自然特例，重复投影自动去重；
- 三个现有实例作为回归测试。
- 私有 POTCAR 拼接/哈希验证、SCF→Band CHGCAR 硬链接交接和显式执行保护；
- SOC/非共线 INCAR 若显式给出 `MAGMOM`，提交前必须能解析为恰好 `3 × NIONS` 个分量；程序只检查维数，不猜磁构型，也不会把这条规则误用于普通共线计算；
- 从真实 EIGENVAL+PROCAR 自动积分轨道权重；
- exact-band 模型的通用 DFT/Wannier 误差、带隙、HR/中心、spread 检查与 PDF；
- RMSE/P95/最大误差只统计 DFT 能量位于项目声明目标能区内的 `(band,k)` 点；一条能带在路径其他位置离开目标能区后不再计入该区间的误差。间接带隙仍在映射后的价带/导带上单独完整检查，避免“缩小绘图区”掩盖错误带隙；
- 单结构/共同基底的最小排名和无 POTCAR export。
- 用 SCF 均匀网格与高对称路径共同判断 exact band block；
- 不可覆盖的候选 attempt、真实执行、失败分类与定向下一动作；
- 所有候选在同一目标能区判分，输出试算优先级/拟合排名/光学基组建议三张逻辑榜。
- 每种投影候选只用其最佳能窗/局域化 attempt 参加最终排名，避免 Top-K 被同一轨道空间的多个重试参数重复占据；因此 Top-K 表示不同的轨道子空间，而不是同一投影的参数微调。
- `common_basis` 从所有结构的 SCF/Band 能量和平均轨道特征共同规划同一 attempt/能窗；联合排名禁止混用不同重试参数，并按最差成员误差排序。
- exact 判定的最小全局边界间隔可配置；InSe 以 0.05 eV 门槛强制进入真实解纠缠分支。
- 已完成 SCF 可在四个科学输入哈希完全一致时导入；Band 还必须核验 CHGCAR，避免复用错误电荷密度。
- MPI/沙箱等基础设施秒退与数值失败分开分类；仅前者允许同一 attempt 一次审计化重试。
- 解纠缠重试是带编号的有限状态机：窗口尝试总数最多 4、局域化重试最多 3；失败只选择尚未执行的同类分支，不改变质量门槛。
- C2DB 只读筛选保留来源 ID/路径、筛选字段和复核标记，不使用会重排 POSCAR 元素或删除旧目录的旧脚本逻辑。
- 支持 `energy_reference: fermi_relative`：候选权重和能窗在统一的 `E-E_F` 坐标中规划，每个结构在写入 Wannier90 前自动转成绝对能量，并同时保存参考计划和物化后计划。
- 新增单命令 `run/resume`、进程级 PID 断点、单 run 锁、有限分支选择和机器可读 `workflow-summary.json`。
- 新增纯 Python `supervise`：阶段边界调度、可配置轮询/心跳、PID 恢复、无进展/超时告警、墙钟截止和信号安全 checkpoint。
- 新增 top-K→MATLAB 响应计划层：自动传递 `nocc_model`、串行作业、PID/指纹断点和旧 MATLAB 活进程附着等待。
- 新增 POSCAR 2D/3D 周期性审计、按倒空间间距生成均匀网格，以及可选 ASE 原生 band-path provider；VASPKIT 仍是待单独验证的可选 adapter，不是核心依赖。
- 新增 `fit-from-bands` 单入口；请求 top-K 与实际通过数量都会显式写入排名/export，不会把不足数量伪装成完整 top-K。

数据集与跨数据库 benchmark 的字段、切分和 ML 路线见 [DATASET_AND_BENCHMARK.md](DATASET_AND_BENCHMARK.md)。

为了防止根据三个已知答案不断补特例，候选规则的开发集/冻结盲测集/外部盲测集和评分方式见 [GENERALIZATION_PROTOCOL.md](GENERALIZATION_PROTOCOL.md)。

下一步只收敛 v1.0：完成可执行 preprocess 状态机、对 PyW90 完整壳层与
`px/py/pz` 分量建议做真实 attempt 回归，并完成 BiTeI/InSe/PtO2 的盲输入回放和
全新材料冷启动。BiTeI/InSe/PtO2 回归已完成；SiS₂ 和 MoOS 均因首次运行促成
通用修正而归入开发集，MoOS v2 已完成、SiS₂ 有限搜索后科学性阻塞。代码与协议
冻结后，AsClS SOC 已作为第一项真正未见样本完成并给出三套合格模型；GeS2 与 MoSSe
现以同一冻结代码分别检验不同层群和 d-p 杂化。下游 shift-current/SHG 保持为独立私有模块，只消费 Stage-1
通过硬门槛的 top-K 包，不是 Stage-1 v1.0 的通过条件。

## 科学边界

能窗“合法”只表示状态数满足数学约束，不表示 Wannier 拟合已经优良。最终候选仍
必须通过插值误差、带隙/连接性、spread/中心、Hermiticity，以及（若请求）多结构
共同基底检查。shift current/SHG 的 top-K 稳健性是下游物理分析，不反向改变本阶段
的候选排名。
