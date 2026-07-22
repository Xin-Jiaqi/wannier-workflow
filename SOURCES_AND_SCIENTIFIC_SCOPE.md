# 材料源、科学问题与 Wannier 扩展用途

## 结构集合不等于铁电双态

工作流输入是一个 `ensemble`，成员数为 1..N：

- `independent`：单材料或多个互不相关材料，各自生成和排名投影。
- `common_basis`：只有比较畴态、堆垛、应变、相变路径或相关构型时才启用；所有成员必须用共同投影/顺序，并按最差成员排名。

L1/L2 只是 `common_basis` 的一个实例，不进入算法假设。

## 高通量入选漏斗

数据库只负责提供结构、来源和初筛元数据，所有正式标签由统一计算设置重新产生。

1. 来源与版本：保留 source ID、数据库 release、结构 provenance 和许可。
2. 去重：规范化晶胞后按结构匹配/哈希去重，同时保留“同源重复”的映射表。
3. 物理初筛：非中心对称、半导体、合理带隙、非金属；按目标可再限制 2D、非磁性、稳定性和最大原子数。
4. 统一复核：用同一结构规范、泛函/POTCAR、SOC 策略和 k 密度重算 SCF/Band；数据库 band gap 只作预筛，不作最终标签。
5. 分层抽样：覆盖 s/p/d、轻/重元素、弱/强 SOC、孤立/纠缠能带、不同点群/原型和单胞大小，避免 benchmark 只剩最容易的材料。

建议来源分工：

- C2DB：首批 2D 计算材料池，属性丰富，适合快速筛选。
- MC2D：来自实验 bulk 的计算剥离结构，具有独立 provenance，并提供 Materials Cloud/AiiDA/OPTIMADE API。
- Materials Project：扩大到 3D 非中心对称半导体和补充化学空间；需记录 API/database version 和非标准 cell 表示。

对于 shift current/SHG，中心对称材料可在生产池中排除；但方法 benchmark 最好保留少量中心对称或金属负对照，用于验证筛选和失败分类，而不是全部只选“预期会成功”的样本。

## 工作的核心意义

“把所有组合都遍历一遍”可以作为小规模 exhaustive oracle，却不应是最终算法：

1. 用轨道覆盖率和化学壳层产生有限、嵌套候选，而不是任意组合爆炸。
2. 用每个 k 点状态数约束先排除不可能能窗。
3. 将失败分成 `not_enough_states`、`too_many_frozen`、`localization_not_converged`、`bad_connectivity`、`quality_gate` 等，再做有方向的有限重试。
4. 同时优化拟合精度、局域性、物理完备性、跨构型可比性和下游 observable 稳定性，输出 Pareto top-K，而不是宣称存在唯一“最好投影”。
5. 在一小部分材料上做 exhaustive search 作为近似 ground truth，量化确定性搜索/后续学习排序节省了多少 Wannier 尝试与人工时间。

这比“力大砖飞”的学术贡献更明确：**可复现的约束搜索、失败知识库、跨结构可迁移基底，以及 observable-aware 质量定义。**

## 除二阶非线性光学以外

同一个高质量 Wannier Hamiltonian 还可用于：

- 高密 k 网格能带、DOS、费米面和有效低能模型；
- Berry curvature、异常霍尔/自旋霍尔、轨道磁化、极化和回旋响应；
- 拓扑不变量、Weyl/Dirac/nodal-line 搜索和表面态；
- Boltzmann 输运、Seebeck/电导/电子热导和 Landauer 弹道输运；
- 电子-声子插值、迁移率和超导相关计算；
- DFT+DMFT/多体模型、磁交换/紧束缚模型、缺陷/界面/莫尔大尺度模型。

因此 benchmark 可以先以 shift current/SHG 为严格下游测试，但软件和数据结构不应写死为“非线性光学专用”。

## 不应建立一个“万能材料池”

不同 Wannier 用途要求不同母池；数据库源可以共享，但物理过滤器必须按任务分开：

| 子池 | 必要性质 | 优先覆盖 | 不必强制 |
|---|---|---|---|
| shift current / SHG | 非中心对称、半导体、结构稳定、目标光谱范围内有带隙 | 极性/Janus/层状、强共价、p/d 轨道混合、轻/重元素与 SOC 分层 | 铁电、磁性 |
| 可切换 BPVE 子池 | 上述条件 + 可定义的极化/堆垛/畴态对 | 能垒合理、两态结构可比、共同基底稳定 | 所有材料都必须有 L1/L2 |
| Berry/AHE | 金属或窄隙体系可保留；AHE 需要破坏时间反演 | 磁性、重元素、SOC、避免交叉/小能隙 | 非中心对称、半导体 |
| spin Hall / topology | 强 SOC、带反转或近简并、相关晶体对称性 | 重 p/d 元素、半金属/窄隙材料 | 磁性、极性 |
| 电子输运/热电 | 稳定结构、可控带边/费米面、足够致密插值 | 低有效质量、多谷、合适带隙和掺杂窗口 | 非中心对称、强 SOC |
| 低能/关联模型 | 孤立或可解释的 d/f 低能流形 | 窄带、磁序、明确局域轨道 | 小 spread 自动等于物理正确 |

第一篇应用工作建议只建立 **NLO 主池**：C2DB + MC2D 为 2D 主体，Materials Project 补充 3D/层状化学空间；先去重，再以统一 PBE(+SOC) 计算复核。初筛字段包括：反演对称性、数据库带隙、原胞原子数、热力学/动力学稳定性、磁性、维数、点群、元素与 SOC 强度。数值阈值应作为版本化配置，不写成算法常数。

## 文章应以物理发现为主线

已有工作已经对 C2DB 的数百种 2D 材料做过 shift-current 效率筛选，也已有更大规模的红外 shift-current 高通量研究；因此“再筛更多材料”本身不够。更有辨识度的主线是：

> **Wannier 表示不确定性如何影响非线性光学材料发现，以及哪些候选在投影、SOC、k 网格和结构扰动下仍保持稳定的大响应？**

建议的物理论文结构：

1. 用自动流程对非中心对称半导体产生 top-K Wannier 表示，不把单一漂亮拟合作为真值。
2. 以 top-K 的 shift current/SHG 离散度定义“表示不确定性”，淘汰能带看似合格但响应不稳定的候选。
3. 给出稳健材料榜单，并分析点群允许张量、SOC、带隙、JDOS、Wannier hopping range、轨道混合和 shift vector/量子几何之间的关系。
4. 对前 10–20 个候选做致密 k 网格、SOC、结构扰动和更高层级带隙校正；对可切换子集用共同基底研究响应翻转或增强。
5. 软件的成功率、失败状态机和计算节省放在方法/补充材料；正文结论落在材料规律、候选与可检验物理上。

这种定位比纯算法更接近 npj Computational Materials：算法是可靠发现物理结论的基础设施，而不是论文终点。数据集和后续学习排序可以作为第二阶段；在获得足够失败/成功 attempts 前，不急于训练复杂模型。

相关先例与方法基线：

- Automated high-throughput Wannierisation (SCDM): https://arxiv.org/abs/1909.00433
- Projectability disentanglement: https://arxiv.org/abs/2303.07877
- C2DB 326 种 2D 材料 shift-current 效率研究: https://www.nature.com/articles/s41524-023-00983-z
- 2D 单质铁电 shift current 的物理设计原则: https://www.nature.com/articles/s41524-023-01026-3
