# 已有体系与自动化回归矩阵

更新：2026-07-22。本表不是材料性能排名，而是说明每个已完成体系为无 AI 流程验证了哪一类必要分支。

| 体系 | 结构/电子特征 | 已有 Wannier 特征 | 对自动化的价值 |
|---|---|---|---|
| BiTeI nSOC | 3 原子极性单层，无 SOC 快速基线 | 9 WF，Bi/Te/I-p，近费米区带块简洁；RMSE 约 0.455 meV | 最小冷启动、exact 带块、无 SOC 维数和快速端到端回归 |
| BiTeI SOC L1/L2 | 两个相关单层构型，重元素 SOC；L1/L2 不是层数 | 18/22/24 WF 共同候选；24-WF all-sp 共同 rank 1，pair-worst RMSE 约 0.093 meV | SOC/spinor `NUM_WANN`、共同轨道顺序、同 attempt/能窗、worst-case 成对排名 |
| InSe SOC L1/L2 | 4 原子 Se–In–In–Se 单层，多带纠缠 | 24/28/32 WF 均可通过；28-WF rank 1，真正 disentanglement；shift current/SHG 对子空间有明显敏感性 | 外窗/冻结窗状态数、有限能窗迭代、“能带合格不等于响应唯一”的核心压力测试 |
| SnS₂ SOC | 6 原子本地双层，s/p 子空间 | 48-WF complete-sp 和 40-WF weight-80 通过；36-WF p-only 失败 | 轨道壳层不完备的淘汰、top-K 稳健对照、Wannier 网格与响应网格分层 |
| PbI₂ SOC | 6 原子重-p 双层 | 36/40/48 WF 三个候选均通过，RMSE 约 0.45–0.48 meV | 多个近乎等价的高质量模型，适合检验排名稳定性和下游响应离散度 |
| PtO₂ SOC | 6 原子 d–p 双层，模型维数较大 | 44-WF Pt-d+O-p 通过，48-WF complete-sp 明显失败，56-WF expanded 通过 | 验证“轨道更多不一定更好”、中间候选失败后继续有限搜索、d–p 纠缠压力测试 |
| C2DB GeSe/SeSn/SSn/GeTe SOC | 2 原子 `p3m1` 独立单层同族，不是双层或常见正交相 | 15×15 rank-1 RMSE 约 0.212/0.223/1.200/0.356 meV；SSn 需 26-WF all-relevant，而其他常用 14-WF 紧凑候选 | 外部数据库 provenance、9→15→21 生成网格、粗网格失败/细网格恢复、紧凑到 expanded 的跨材料分支 |

## v1.0 最小验收集

1. **BiTeI nSOC**：最小快速冷启动。
2. **BiTeI SOC pair**：共同基底与 worst-case 排名。
3. **InSe SOC pair**：必须进入真实 disentanglement。
4. **PtO₂ 或 SSn 已有输出回放**：必须保留失败候选并进入后续候选，不因首次失败整个中止。

## 响应层应如何利用这些体系

- 每个通过硬门槛的 top-K 都产生独立 shift-current/SHG 作业，`NOCC` 只读取 `nocc_model`。
- 单材料先比较相同结构的不同子空间：全谱 L2、峰值/峰位、符号、积分强度、张量分量。
- pair 体系另外比较 L1/L2 的对应分量；不得为两个结构选用不同的 rank/轨道顺序。
- 跨材料统计只在单材料 top-K 分析完成后进行，并按单层/双层、s/p/d–p、SOC 强度和结构原型分层。
