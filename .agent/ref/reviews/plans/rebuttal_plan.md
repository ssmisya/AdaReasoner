# GenReasoner 期刊 Rebuttal 方案（2026-08-04 审计版）

> 交付物：point-by-point response PDF + 修订 manuscript。任何数字必须绑定配置、结果文件和评分协议；不得把计划、smoke run 或预期值写成完成结果。

## 0. 总体判断

当前 rebuttal **已有主体证据，但尚不可提交**。优势是：固定 checkpoint 的多次随机推理、GUIChat 72B judge、500 条 judge semantic audit、阶段时延和 early-fault accuracy 已形成可复核结果。主要风险是：WebMMU 产物未同步、HRBench 评分器回退污染、E4 无 no-tool 样本、E5 detect 指标未校准，以及 judge 的作者/第二标注者确认、matched-budget 曲线和近重复检查未完成。

## 1. 会议版与期刊增量（R1-1 / R2-1）

会议版：*AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning*，ICLR 2026，OpenReview `nUGPEmQ2ut`。

继承内容：轨迹构造、Tool-GRPO、composite reward、七工具套件、单任务结果。

期刊增量：

1. identifier randomization + description paraphrasing 的 interface robustness；
2. randomized cold-start/RL study；
3. V\*/HRBench tool-planning evaluation。

**动作：**在 Abstract、Introduction、Contributions、Conclusion 和表格 caption 中统一该边界，不把会议版结果重新声明为新贡献。

## 2. 泛化 claim（R1-2 / R2-2）

- “new tasks”改为 **cross-stage transfer**：final policy 在 Tool-GRPO 阶段见过任务。
- “new tools”改为 **interface-level robustness**：名称、描述、schema 或顺序变化，但功能不变。
- 不声称 zero-shot new task family、abstract function understanding 或 genuinely novel capability。

E1 不再作为硬实验执行；采用 claim narrowing。

## 3. 数字可靠性（R2-3）

已可用的固定-checkpoint inference repeats：VSP 89.27±0.88、VSPO 78.64±0.33、Jigsaw 88.27±0.12、BLINK-J 88.22±0.39、V\* 68.06±0.53、GUIChat 72B judge 73.60±0.11。

必须明确：这些是 stochastic inference repeats，不是 training-seed variance，也不是显著性检验。

阻塞项：

- WebMMU 71.75±0.53 仅由执行日志记录；当前副本缺三次 full/judged artifacts。
- HRBench 63.04 含每 run 108–111 个 `Z` 回退；明确答案重提取仅给出 ≥68.92±0.14 的下界，终值需统一重判。

## 4. Reward（R1-5）

现有结果不支持 cost-aware abstention：VSP/Jigsaw 近全量用工具，GUIChat 三次均 962/962 样本至少调用一次工具。难度与调用数相关，但不能因果归因于 asymmetric reward。

**策略：**删除“正确且不调用工具”实证 claim；只保留 bounded reward shaping 和无额外 convergence proof。若要保留 selective-use claim，必须补 easy/no-tool 对照及 symmetric/asymmetric reward ablation。

## 5. 成本与时延（R1-6 / R2-5）

正式 JSON 口径：Jigsaw generation/tool=90.95/0.55%，VSP=39.11/59.42%；AStar/Point=0.092/255.333 ms，约 2,775×。

这只支持“CPS 不是成本 proxy”。Reviewer 要求的 matched-budget accuracy–latency curve 仍缺，不能用 stage totals 代替。

## 6. 故障鲁棒性（R1-6c / R2-4）

已有 VSP/Jigsaw 各 100 条、五类 early fault。可信结论是 accuracy delta：VSP 最差 timeout −6pp，Jigsaw 最差 missing −17pp。

当前 detect/react 启发式把 fault 后任意工具调用计为检测，clean baseline 也为 1.0。必须人工校准或删除 detect/propagate 数字。Late injection 和 with/without failure-reflection 训练消融未完成，不做训练归因。

## 7. Baseline 公平性（R1-3 / R1-4）

- 明确 proprietary main-table rows 是 no-tool、single-turn。
- 保留 GPT-5+Tools matched protocol：VSP 55.64→71.36，Jigsaw 80.10→84.50。
- DeepEyes/Pixel-Reasoner 未适配 multi-tool interface；只能解释为 unseen-interface brittleness，不能解释为 inherent inferiority。

## 8. Judge、泄漏与复现（R2-6 / R2-7 / Minors）

- 已完成 500 条 judge semantic audit：498 条有效，agreement=90.76%（Wilson 95% CI 87.90%–93.00%），κ=0.781，FP/FN=15/31；GUIChat/WebMMU agreement=86.29%/93.69%。
- 已完成描述性长度四分位检查；最长 quartile 的 agreement 为 91.20%，未呈单调优势。若正文声称 human validation，需作者盲审确认或第二独立人类标注，并可补控制 correctness 的正式长度回归。
- 在完整 Jigsaw train/test source images 上运行 pHash Hamming≤5 + CLIP cosine≥0.95，报告真实 overlap/flagged pairs。
- 核对 332,649 cold-start 样本按任务/阶段分解。
- 统一模型命名、网格大小、任务数、表引用、Tool-GRPO 术语及 syntactic/semantic success 定义。

## 9. 提交顺序

1. 同步 WebMMU full/judged artifacts；统一重判 HRBench `Z`。
2. 决定 judge 口径：称 single-reviewer semantic audit，或补作者/第二人类标注者确认；可选补正式长度回归。
3. 生成 matched-budget accuracy–latency curve。
4. 校准 E5；决定是否承担 late fault / 训练消融。
5. 运行 pHash+CLIP；核对 332,649 分解。
6. 将引用、真实 section/table/figure 编号和 limitations 落入 manuscript。
7. 对照 `RESULTS_TABLE.md` 做最终数字审计并生成 point-by-point PDF。
