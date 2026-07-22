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

当前 `04_python` 的 shift-current/SHG 实现已经可以读取前三个结构文件。它与 W90Flow 保持为独立模块；连接层只读取 `package-manifest.json`，不复制 Wannier 候选逻辑，也不反向改变拟合排名。

第二模块的接口规则为：

- 二维结构默认 `71×71×1`；三维结构默认 `51×51×51`。
- `NOCC` 直接读取 `package-manifest.json:nocc_model`，不再手工修改。
- 从 POSCAR/项目维数得到空间群操作，按 Neumann 原理求三阶张量的不变量空间，自动列出允许分量、必须为零的分量和彼此相等/反号的分量；shift current 与 SHG 分别使用各自的末两指标交换规则。
- 光子能量范围和展宽可以作为统一 profile 默认值，用户只有在做收敛研究时才覆盖。

当前实现状态要区分清楚：`NOCC=nocc_model` 的读取和硬校验已经完成；`71×71×1 / 51×51×51` 的自动默认、POSCAR 对称性张量生成和纯 Python 响应核的连接属于下一独立补丁。在补丁完成前，响应计划仍要求显式提供 `kmesh` 和 `components`，不能把文档中的设计写成已完成能力。

二维“有效厚度”是唯一不能仅由对称性决定的量：含真空的晶胞高度是人为设置，不能直接当材料厚度。稳妥做法是优先同时输出与厚度无关的 sheet response；如需换算成体响应，再由项目明确采用几何层厚、实验层间距或其他统一约定。

当前物理边界仍需保留：该响应实现使用二维 `k_z=0` 和 diagonal tight-binding approximation。把代码接入同一仓库不会自动消除这个近似，因此在完整位置矩阵验证前保持独立更安全。
