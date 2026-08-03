# GenReasoner 期刊 Rebuttal 方案(point-by-point)

> Springer Major Revision。交付物 = point-by-point response PDF + 修订 manuscript(无 tracked changes)。
> 两位 reviewer:R1(6 点)、R2(7 major + minors)。收敛于 A(与 ICLR 划界)+ B(泛化超证据)。

## 0. 继承 vs 新增(delta 地基)
ICLR 三贡献 = ①多轮工具规划的数据构造 ②adaptive RL(Tool-GRPO)③轻/重工具套件;+ AdaReasoner SOTA + "超过 GPT-5/Claude"。
**期刊 headline 几乎照搬。** 真正新增(delta):
- (1) 标识符随机化 + 改述的 Adaptive Learning(期刊 Sec 2.4)—— ICLR 无此章节
- (2) Rnd TC + Rnd TG 泛化研究(期刊 Table 4)
- (3) V*/HRBench 工具规划对比(期刊 Tables 5-6)
继承(已发表,须归到 ICLR 名下、别当新卖点):轨迹构造、Tool-GRPO、reward、7 工具、Table 2 单任务 SOTA、GPT-5 对比、emergent 行为。

## A. 与 ICLR 划界〔R1.1 + R2.1〕—— 纯写作,最高优先
动作:① 正文显式引用 ICLR 2026;② 重写 contribution,把 headline 从"单任务 SOTA"移到 delta;③ 评估重心压到 delta。
可直接用的 contribution 改写草案:
"This work extends our conference paper (AdaReasoner, ICLR 2026 [cite]). The conference versio n established the trajectory-curation pipeline, the Tool-GRPO algorithm, the reward design, and the seven-tool suite, achieving single-task SOTA (Table 2). This journal extension makes three *new* contributions: (i) an identifier-randomization and description-paraphrasing Adaptive Learning method that yields interface-robust tool use (Sec 2.4); (ii) a systematic generalization study under randomized cold-start and RL (Rnd TC + Rnd TG, Table 4); and (iii) a tool-planning evaluation on V* and HRBench (Tables 5-6). We restate inherited results only as background."

## B. 泛化超证据〔R1.2 + R2.2〕—— delta 核心,复用+补一个实验
- 收窄 claim:把"新任务 zero-shot"改述为"跨阶段 transfer"(承认三任务数据在 Tool-GRPO 见过);把"新工具"改述为"接口级鲁棒性(标识符/描述变化,功能不变)"。
- 复用:Yfxj(ICLR rebuttal)已构造感知/操作类新工具,证明"选择性使用"(用有用的、忽略无关 RotateImage=0、弱化冗余)。可作为"工具选择判断力"证据。
- 缺口→需补 1 个实验:引入一个"提供模型缺失能力、且被某个从所有阶段(TC+TG)都排除的任务真正需要"的新工具,证明 zero-shot 有效使用(不是冗余/无关)。这是唯一能正面顶住 R2.2 的硬证据。

## C. baseline 公平/闭源透明〔R1.3 + R1.4〕—— 大部复用
- 复用 W5WP(ICLR rebuttal)Q1/Q2/Q3:闭源模型如何评、是否一轮作答、用多步框架重评。→ 在正文补"闭源模型同工具集/同 prompt/同多步协议"的说明表,回应"超 GPT-5 是否公平"。
- 补:DeepEyes/PixelReasoner 适配说明——它们为单工具/固定循环设计,说明我们做了何种最小适配;若不适配则明确标注,并解释 Table 6 低分含 prompt 不兼容成分,避免"贬低 baseline"读感。x

## D. reward 理论 + hacking〔R1.5〕—— 辩 + 自查实验
- 辩:非对称设计是"结果优先、工具为辅"的显式意图,不是漏洞;不硬造收敛定理(GRPO 收敛性沿用原文),改用"经验稳定性"回应。
- 补:reward-hacking 自查——统计"答对样本中未调用工具的比例"及其正确率;做对称 vs 非对称 reward 的消融(是否损害该用工具的难题)。若自查显示未 hacking,直接摆数据。

## E. 成本/延迟〔R1.6 + R2.5〕—— 必须新测(0 复用)
测:latency 分布、吞吐、gen/exec/orch 三段拆分、per-tool 时间;画"匹配预算下 精度–延迟曲线"vs baseline。强调 CPS≠成本(本地算子 vs 专家模型差数量级)。

## F. 失败鲁棒性〔R1.6c + R2.4〕—— 必须新补
受控故障注入(错误但合理输出/缺失/畸形/超时/矛盾;早注入 vs 晚注入),测 检测/忽略/恢复/传播;补"有/无 failure&reflection 轨迹"的消融。

## G. 跨表数字打架〔R2.3〕—— ⚠️硬伤,先查后改
硬 bug:裸 Qwen2.5-VL-7B GUIChat 59.46(T2) vs 68.09(T4/5);3B WebMMU 55.89 vs 54.47;GUIChat 45.11 vs 46.26。
动作:定位两套数字的评测条件差异→统一口径重报→每张表标注评测设置;关键表补 multi-seed(≥3 seeds,mean±std)。**不查清不能提交**。

## H. Jigsaw-COCO 泄漏〔R2.6〕—— 核实+改split或声明
核实 C.1:是否同图 3 patch 训、第 4 patch 测(=留位置非留图)。→ 改 image-disjoint(或 COCO-val)+ 近重复检查;至少在文中明确 disjointness 保证。

## I. LM-judge 验证〔R2.7〕—— 新补
Qwen2.5-VL-72B 裁判(V*/WebMMU/GUIQA):补 人-机一致性定量研究(样本量/标注人数/一致性/盲评)+ "冗长答案不占便宜"的检验。

## Minors〔R2〕
命名统一 GenReasoner(图1/10、repo 都改,别让它像 baseline);加 Limitations 章节(延迟成本/依赖手工轨迹与外部专家模型/任务专用工具/工具质量是双刃);软化两处过度声称("never explicitly trained"、"bottleneck 从 scale 到 tool quality"限定为结构化任务);释放轨迹/split/seed;332,649 冷启样本给分解;VSPO 网格、任务数(4 vs three)、表引用、术语(Tool-GRPO)、语法成功 vs 语义有用 —— 逐条改。

## 工作量三分
- 纯写作/硬改(0 实验):A、G(对齐)、H(声明)、命名、Limitations、minors、软化 claim
- 复用已有(改写即可):B(部分)、C(大部)、复现/超参(W5WP Q4-6)
- 必须新跑实验:E(成本)、F(故障)、G(multi-seed)、I(judge 一致性)、B(补 1 个真新能力工具实验)、D(hacking 自查)

## 建议推进顺序
1) 先清"不需实验、最锋利"的刀:A(delta 改写)+ G(数字对齐)+ 命名 + Limitations —— 直接消掉"restate published""numbers don't reconcile"两条致命指控。
2) 再复用 B/C 已有弹药改写。
3) 最后排新实验:优先 G-multiseed、E-成本、F-故障、B-新能力工具、I-judge、D-hacking(按算力/时间排期)。
