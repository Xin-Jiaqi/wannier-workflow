# Wannier → 非线性光学的文件接口

W90Flow v1.0 只负责产生和排序 Wannier 模型，不把 shift current/SHG 的物理求和器并入核心包。

每个通过质量门槛的 rank 交付以下文件：

- `POSCAR`
- `wannier90_hr.dat`
- `wannier90-wcenter.dat`
- `wannier90_centres.xyz`
- `wannier90.win`
- `quality.json`
- `package-manifest.json`

下游程序只需读取交付目录。`package-manifest.json` 中的 `nocc_model` 是完整 hr 模型的占据带数，不能使用只为公共拟合能区定义的 `nocc_target`。

当前 `04_python` 的 shift-current/SHG 实现已经可以读取前三个结构文件。以后只需增加一个小 adapter：从 `package-manifest.json` 自动取得输入目录和 `nocc_model`，然后逐个 rank 运行。k 网格、光子能量范围、展宽、张量分量和二维有效厚度属于响应计算设置，不属于 Wannier 搜索，也不阻塞 W90Flow v1.0 发布。

当前物理边界仍需保留：该响应实现使用二维 `k_z=0` 和 diagonal tight-binding approximation。把代码接入同一仓库不会自动消除这个近似，因此在完整位置矩阵验证前保持独立更安全。
