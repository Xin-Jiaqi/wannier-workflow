# W90Flow v1.0 无 AI 自动化规格与验收门槛

更新：2026-07-22。本文档冻结 v1.0 的范围，避免在材料池、响应计算和软件开发之间无限扩张。

## 1. 定义

“纯 Python、无 AI”指所有流程控制、候选生成、能窗搜索、失败分类、恢复、质量门槛和排名都由确定性 Python 代码完成，不需要 AI 在运行中选投影或调参。VASP 和 Wannier90 仍是外部数值引擎；Python 不重写 DFT 或 MLWF 算法。

PyW90 在 v1.0 中是可选 baseline/adapter，不是必须依赖。主流程必须能独立从 `EIGENVAL + PROCAR + POSCAR` 生成可审计的投影与能窗候选。

## 2. 输入和输出

### 必需输入

- 一个或多个已优化 `POSCAR`；不假定存在 L1/L2。
- `project.yaml`：SOC、能量参考、目标能区、`independent/common_basis`、候选/排名上限和硬门槛。
- SCF、Band/PROCAR 和 VASP–Wannier 接口的 INCAR/KPOINTS 模板。
- 本地私有 POTCAR registry 及 SHA256 lock；永不进入公开导出。
- 已配置的 VASP/Wannier90 启动器。

### 关键交接

| 阶段 | 最小输入 | 必须输出 | 硬检查 |
|---|---|---|---|
| prepare | POSCAR、模板、POTCAR lock | 输入快照、哈希、私有 POTCAR | 元素顺序与组合哈希 |
| SCF | POSCAR/POTCAR/INCAR/KPOINTS | CHGCAR、OUTCAR、EIGENVAL | 电子收敛、文件完整 |
| Band | CHGCAR + Band 输入 | EIGENVAL、PROCAR、vasprun.xml | band/k 点数和文件完整 |
| Candidates | POSCAR、EIGENVAL、PROCAR | `candidate-spec.json` | SOC 维数、轨道覆盖、模型上限 |
| Attempts | 候选 + SCF/Band 能量 | 有限 exact/disentanglement 计划 | 外窗状态数、冻结窗、包含关系 |
| Wannier | 不可覆盖 attempt 输入 | wout、hr、centres、band.dat | 正常结束、维数、Hermiticity |
| Quality | DFT/W90 bands + hr/centres/wout | `quality.json` + cmp PDF | 统一目标区误差、带隙/连接性、spread/中心 |
| Rank/Export | 所有通过和失败项 | top-K + manifest | hard gates 先于排序，pair 用 worst-case |

### 最终交付

- 每个结构或共同基底的 top-K 目录；
- `POSCAR`、`wannier90_hr.dat`、`wannier90_centres.xyz`、纯中心文件、`wannier90.win`、比较图、`quality.json`、`manifest.json`；
- 完整的通过/失败 attempt 数据和失败类别；
- 不含 POTCAR、CHGCAR、MMN/AMN 等大型或许可文件。

## 3. 有限状态机

```text
prepare → SCF → Band/PROCAR → candidates → finite attempts
       → quality hard gates → ranking → top-K export
```

对每个投影候选只允许有限分支：

1. exact band block（只在目标带块全局隔离时），否则 disentanglement seed；
2. 收敛失败时最多一个 `convergence_retry`；
3. 状态数、窗口或能带质量问题时最多一个 `outer_window_expansion`；
4. 基础设施中断只允许同输入原地恢复一次；
5. NaN、段错误或未能安全分类的数值失败立即 `BLOCKED`；
6. 不会自动放宽能带误差、带隙、连接性、spread 或中心硬门槛。

一个候选穷尽后转向下一个投影；所有候选穷尽仍无合格项时，输出可解释失败而不是无限调参。

## 4. v1.0 必须通过的验收

- `prepare` 后一条 `run --execute` 能进入完整链；`resume --execute` 使用同一状态机。
- dry-run 不写科学状态；未 prepare 的 run 不得启动计算。
- 同一 run 不得有两个监督器或重复 VASP/Wannier 进程。
- 每个长进程记录 PID；中断后能根据完整输出回收，或审计化恢复一次。
- 纯 Python 长时监督器按可配置短间隔轮询 PID/输出，定时写状态心跳，并在完整阶段边界自动衔接下一阶段。
- 墙钟截止、`SIGINT/SIGTERM`、阶段超时和无进展检测必须保存 checkpoint；只告警/安全阻塞，不得自动修改科学输入或放宽门槛。
- 单结构、多结构 independent 和 common-basis 都可用；共同基底不得混用不同 attempt/能窗。
- 支持 exact 和真实 disentanglement；SOC 自动维数必须正确。
- 回归夹具通过 BiTeI nSOC、BiTeI SOC pair 和 InSe SOC pair；InSe 必须真正走 disentanglement 分支。
- 完成一次冷启动端到端测试、一次人为中断/恢复测试，且 top-K/哈希结果可复现。
- PyW90 对照输出不得改变硬门槛；与内部候选不一致时同时保留两者供审计。
- 单元/集成测试、CLI 帮助、版本化 schema、一页用法和失败说明齐全。

## 5. 明确不属于 v1.0

- shift current/SHG 长时生产计算及物理结论；
- 材料池大规模扩展、机器学习、GUI 或自主 AI agent；
- 无物理目标地自动决定唯一的目标能区；目标能区必须在项目配置中声明；
- 用 WannSymm 或放宽门槛修饰一个不合格的 Wannier 模型。

## 6. 开发顺序

1. 单命令监督器、进程锁、有限重试和 resume；
2. 离线模拟端到端状态转移测试；
3. PyW90 结构化对照 adapter；
4. BiTeI/InSe 已有输出回放和一次小型冷启动；
5. 冻结 v1.0 后才用它扩展本地候选和 C2DB 材料池。
