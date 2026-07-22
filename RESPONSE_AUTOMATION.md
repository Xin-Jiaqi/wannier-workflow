# Top-K → MATLAB 响应计算自动化

这一层只负责把已通过 Wannier 硬门槛的 top-K 包编成有序响应作业，不修改 MATLAB 科学核。计划的默认模式是 `dry-run`；只有明确加 `--execute` 才会启动 MATLAB，并且始终单作业串行。

## 占据数的唯一规则

- `quality.nocc_target`：只是公共目标能区拟合判分时的占据带数。
- `manifest.nocc_model == quality.nocc_model`：完整 hr 模型的占据带数，必须作为 shift current 和 SHG 的 `NOCC`。

计划生成时会硬校验这一关系，同时校验 `quality.passed`、`checks.hr_centres_match`、HR 维数、中心数、必需文件和 manifest SHA256。

## 输入配置

用一个 JSON 冻结每次 campaign：

```json
{
  "schema_version": 1,
  "system": "InSe",
  "export_summary": "../common-top-k-v04/export-summary.json",
  "plan_dir": "./response-plan-InSe",
  "output_root": "../response-runs",
  "matlab_executable": "matlab",
  "response_order": ["shift_current", "shg"],
  "responses": {
    "shift_current": {
      "enabled": true,
      "code_dir": "../02_shift_current/code",
      "entry_point": "compute_shift_current_model",
      "profile": "screening",
      "components": ["xxy", "yxx", "yyy"],
      "kmesh": [31, 31, 1],
      "energy_grid_eV": {"min": 0.0, "max": 3.5, "points": 501},
      "broadening_eV": 0.05,
      "thickness_A": 5.429,
      "parallel_workers": 4
    },
    "shg": {
      "enabled": true,
      "code_dir": "../03_shg/code",
      "entry_point": "SHG_unified_optimized",
      "profile": "screening",
      "components": ["112", "211", "222"],
      "kmesh": [11, 11, 1],
      "energy_grid_eV": {"min": 0.0, "max": 3.5, "points": 301},
      "spectral_broadening_eV": 0.05,
      "effective_thickness_A": 5.429
    }
  }
}
```

## 本地命令

```bash
python -m w90flow.response_plan build campaign.json
python -m w90flow.response_plan preview response-plan-InSe
python -m w90flow.response_plan status response-plan-InSe
python -m w90flow.response_plan run response-plan-InSe
```

上述 `run` 仍然只预览。经人工复核网格、能量网格、厚度和张量后，才能使用：

```bash
python -m w90flow.response_plan run response-plan-InSe --execute
```

`response-plan.json` 固化全部输入与顺序；`response-status.json` 记录 `PENDING/RUNNING/SUCCEEDED/FAILED`。每个作业有独立 `matlab-config.json`、日志和指纹成功标记。启动后还会写入 MATLAB PID；如 Python runner 意外中断而 MATLAB 仍活着，恢复命令会先附着等待原 PID，绝不重复启动。死 runner 遗留的 JSON 锁会保留为 `stale-*` 审计文件。恢复时只跳过指纹一致的成功作业，不会因为目录里恰好有旧文件就误判完成。

## 已有 MATLAB 接口审查

现有 InSe/SnS2 串行脚本的运行规则一致：

- shift current 核 `compute_shift_current_model(settings)` 需要 `model_dir/output_dir/nocc/components/nkx,nky,nkz/ksi_eV/photon_energy_eV/thickness_A/parallel_workers`；
- SHG 核 `SHG_unified_optimized(cfg)` 需要 `input_dir/output_root/seedname/poscar/tensor_components/nk/photon_energy_eV/nocc/spectral_broadening_eV/effective_thickness_A`；
- 两者都从每个模型目录读 `wannier90_hr.dat + wannier90-wcenter.dat + POSCAR`；
- 旧串行脚本的外层 project/profile 接口是为 InSe/SnS2 硬编的，新 backend 因此直接调用其已有科学核，而不复制或改写公式。

## 真正一键执行前仍缺的冻结参数

对新材料，top-K manifest 并不包含下列科学选择，Python 不会猜测：

1. 对称性允许且需要计算的 shift-current/SHG 张量分量，以及坐标轴与晶格方向的映射。
2. screening/production 的 k 网格、光子能量范围/点数、shift-current 宽化和 SHG 谱宽化。
3. 每种材料的有效厚度数值与统一定义（几何厚度、层间距或其他规范）。
4. SOC/非 SOC 下的自旋简并因子与符号约定。当前 shift-current 核在现有 SOC 项目里固定为不额外乘 2；SHG 旧基线固定 `spin_degeneracy=1, sign_factor=1`。
5. MATLAB 可执行文件/许可证环境、Parallel Computing Toolbox 可用性，以及每台机器允许的 workers 上限。
6. 响应结果的独立 full-r/其他实现交叉验证门槛。当前两个 MATLAB 核都是 `hr + centres` 的 diagonal-TBA 路径，不应自动升格为绝对第一性原理结论。

另外，旧 SHG 核的 `metadata.nocc_physical_status` 仍写为 `unverified`。新 backend 已从 manifest/quality 硬校验 `nocc_model`，但如需让 MATLAB 结果内部也显式记录“已验证”，仍需旧 MATLAB 接口新增 provenance/status 字段；本次没有修改该科学代码。
