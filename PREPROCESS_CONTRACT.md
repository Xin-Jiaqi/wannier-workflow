# Stage 0：POSCAR → Relaxation → SCF → Band

本模块是独立的确定性前处理原型。它只定义科学配置、输入/输出契约、严格后检查和可恢复状态；默认不启动 VASP，也不会因失败自动修改门槛。

## 为什么单独实现

主流程当前最重要的部分仍是 `Band → projection candidates → Wannier top-K`。前处理应通过一个窄接口接入，而不应把结构优化的科学选择混入 Wannier 候选排序：

```text
POSCAR + preprocess protocol
  → relaxation/CONTCAR（可选）
  → SCF/POSCAR + CHGCAR
  → Band/EIGENVAL + PROCAR + vasprun.xml
  → 既有 Stage 1 Wannier 自动化
```

## 必须由用户/项目配置明确决定的内容

- `dimensionality`: 只能明确为 2 或 3；本模块不按材料名称猜测。
- `soc`: 可给一个布尔值，或逐阶段给 `relaxation/scf/band` 三个布尔值。
- `relaxation.enabled`: 是否优化结构。
- `relaxation.cell_mode`: `ions_only`、`full_cell` 或 `constrained_cell`。
- 2D 优化的 `vacuum_axis`。
- 改变晶格时的应力门槛和被检查的应力分量。
- `band.path_provider`: `template/seekpath/ase/vaspkit` 之一。这里记录提供者，不假设某个 VASPKIT 任务号。

对于普通非磁性 PBE/PBE-like 测试，`standard_semilocal_preprocess()` 可在结构周期性已经解析后，把用户输入缩减为主要的 `soc=true/false`：它固定 `EDIFF=10^-6 eV`、`NELM=120`、固定晶格离子优化、`0.01 eV/Å` 力门槛和 ASE 路径。这个命名 profile 不会猜测 Hubbard U、杂化泛函、vdW、磁序或 POTCAR 变体；这些超出 profile 的体系仍必须显式配置。

2D 不会默认放松真空轴：

- `ions_only` 固定三个晶格矢量；
- `constrained_cell` 必须把真空轴放进 `locked_lattice_axes`；
- `full_cell` 只有在显式写出 `allow_vacuum_axis_relaxation: true` 后才允许。

## 严格收敛证据

`analyze_preprocess_outputs()` 同时检查：

- OUTCAR 是否正常结束、是否含 fatal marker；
- OUTCAR 实际 `EDIFF/NELM/NSW/EDIFFG/ISIF/ICHARG` 是否与冻结协议相同；
- 每个离子步的电子循环是否达到 EDIFF，OSZICAR 电子步是否超过 NELM；
- 优化是否出现离子收敛标记；
- 最后一个 `TOTAL-FORCE` 块的最大原子力；
- 改变晶格时指定应力分量的最大绝对值；
- CONTCAR 是否完整、元素/原子数是否不变、锁定晶格轴是否未改变；
- SCF 是否形成非空 CHGCAR；
- Band 是否为固定电荷契约；EIGENVAL 可解析；PROCAR 头中的 k 点/能带维数与 EIGENVAL 一致；vasprun.xml 具有完整根节点结束标记。

Band 是 `ICHARG=11` 的固定电荷计算，因此不把“出现 EDIFF 中止标记”当作它的收敛定义；它要求正常结束和完整交接文件。

## 断点与恢复

`prepare_preprocess_run()` 创建三个阶段目录，冻结协议，物化标准输入并写入 SHA256。INCAR 模板中的强制标签会被协议值确定性替换或追加；同一强制标签出现两次时直接拒绝。KPOINTS 必须是显式模板，命令必须是字符串列表而不是 shell 字符串。

POTCAR 没有任何猜测或自动下载：调用者必须提供以下二者之一：

- 已组合 POTCAR 的绝对路径、POSCAR 元素顺序和预期 SHA256；
- 按 POSCAR 元素顺序排列的组件路径及各自 SHA256，并给出组合文件 SHA256。

POTCAR 只进入本地 run 的 `private/` 和各 VASP 阶段，不进入 Band/Wannier 导出包。

`begin_stage()` 使用原子阶段锁，记录 launcher PID、worker PID、命令、输入哈希和 attempt。行为如下：

- 活 PID：拒绝重复提交；
- 同主机死 PID：只能显式调用 `recover_interrupted_stage()`，然后以完全相同的协议和输入恢复；
- 输入哈希或协议哈希改变：拒绝当作恢复；
- 后检查失败：状态为 `FAIL`，禁止自动重试或自动放宽力、应力、EDIFF 等门槛；
- 已 `PASS`：拒绝重复提交。

这一区分了基础设施中断与科学/数值失败。前者可按相同输入恢复，后者必须进入人工或上层有限策略审查。

`run_preprocess_workflow()` 是同步有限状态机：默认 `execute=False`，只报告下一步且不改变状态；显式 `execute=True` 后按 Relaxation（可选）→ SCF → Band 串行运行。外部进程可随时用 `inspect_preprocess_run()` 看到活动 PID 并等待。若等待器被中断但 worker 仍活着，唯一阶段锁会转交给 worker，禁止重复提交；worker 结束后需显式 `recover_interrupted_stage()`，再用 `reconcile_preprocess_stage()` 验收已有输出或以相同输入恢复。

每个 PASS 阶段只允许以下哈希交接：

- Relaxation：`CONTCAR → SCF/POSCAR`；
- SCF：`POSCAR + CHGCAR → Band`；
- Band：生成 `band-handoff.json`，锁定 `POSCAR/OUTCAR/EIGENVAL/PROCAR/vasprun.xml`，供现有 `fit-from-bands` 接口消费。

## Python 接口

```python
from w90flow.preprocess import (
    normalize_preprocess_config,
    prepare_preprocess_run,
    run_preprocess_workflow,
)

protocol = normalize_preprocess_config(project, dimensionality=2)
run_dir = prepare_preprocess_run(
    "runs/example",
    "POSCAR",
    protocol,
    templates={
        "01_relax": {"incar": "templates/INCAR.relax", "kpoints": "templates/KPOINTS.relax"},
        "02_scf": {"incar": "templates/INCAR.scf", "kpoints": "templates/KPOINTS.scf"},
        "03_band": {"incar": "templates/INCAR.band", "kpoints": "templates/KPOINTS.band"},
    },
    potcar={
        "path": "/licensed/locked/POTCAR",
        "element_order": ["In", "Se"],
        "sha256": "<64-hex>",
    },
    commands={
        stage: ["/absolute/run-vasp", "{stage_dir}"]
        for stage in ("01_relax", "02_scf", "03_band")
    },
)

# Read-only preview.  No subprocess is started.
preview = run_preprocess_workflow(run_dir)

# Execution is an explicit second action.
result = run_preprocess_workflow(run_dir, execute=True)
```

最短命令行只生成并审计协议，不启动计算：

```text
w90flow preprocess-profile POSCAR --soc
w90flow preprocess-profile POSCAR --no-soc --no-relax
```

完整命令行先物化、再预览，最后才显式执行：

```text
w90flow preprocess-prepare POSCAR preprocess.yaml --run-dir runs/material-001
w90flow preprocess-inspect runs/material-001
w90flow preprocess-run runs/material-001
w90flow preprocess-run runs/material-001 --execute
```

Band 通过后无需手抄路径；用优化后的 POSCAR 重新解析周期性和网格，直接进入
Wannier top-K：

```text
w90flow fit-from-preprocess wannier-project.yaml \
  --preprocess-source material-001=/absolute/runs/material-001 \
  --runs-root /absolute/wannier-runs

# 核验 dry-run 无误后才正式运行
w90flow fit-from-preprocess wannier-project.yaml \
  --preprocess-source material-001=/absolute/runs/material-001 \
  --runs-root /absolute/wannier-runs --execute
```

等待器中断后，只有确认原 worker PID 已死亡才可清理锁；如果输出已经完整，先
`reconcile`，不完整时再以同一冻结输入 `run --execute`：

```text
w90flow preprocess-recover runs/material-001 02_scf
w90flow preprocess-reconcile runs/material-001 02_scf
```

## 尚未自动化的边界

- 不选择或下载具体赝势，也不分发 POTCAR；上层必须提供已锁定、已授权的路径/recipe。
- 不替用户决定泛函、U、vdW、磁序、偶极修正、结构约束或 k 路径物理约定。
- `constrained_cell` 只冻结并检查约束契约；VASP 的具体 `ICONST`/模板实现必须由项目验证。
- VASPKIT 目前只作为可选路径提供者名称。接入前要冻结版本、任务号、非交互输入和输出哈希；不应成为核心流程的唯一依赖。
- 当前执行器是同步的；需要作业调度器时，应把同一命令列表封装为 scheduler adapter，而不是改变科学输入状态机。
- 真实 VASP 输出格式还需要用 2D/3D、SOC/nSOC、不同 VASP 版本的脱敏 fixtures 做回归测试。
