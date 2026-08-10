# AdaReasoner Rebuttal — 实验结果与证据状态

> 模型：AdaReasoner-7B-Randomized（Qwen2.5-VL-7B）。结构化任务使用完整任务工具集；开放式任务使用 Qwen2.5-72B-Instruct judge。
> 下表的 run1/2/3 是固定 checkpoint、temperature=0.7 的三次独立随机推理重复；配置未固化显式 `seed:`，因此只能报告 inference-repeat variance，不能称 training-seed variance。

## 一、准确率与证据等级

| Benchmark | 评分协议 | run1 | run2 | run3 | mean±sample std | 当前证据状态 |
|---|---|---:|---:|---:|---:|---|
| **VSP** | gym 路径验证；修复 AStar 等价参数格式 | 89.91 | 89.64 | 88.27 | **89.27 ± 0.88** | 本地 3 个 full result 可核查；500 个 verify/run 均使用 boxed/output/direct 格式，未触发宽松 `"no"` 子串回退 |
| **VSPO** | gym 路径验证 | 78.98 | 78.32 | 78.62 | **78.64 ± 0.33** | 本地 3 个 full result 可核查 |
| **Jigsaw-COCO** | 选项匹配 | 88.20 | 88.20 | 88.40 | **88.27 ± 0.12** | 本地 3 个 full result 可核查 |
| **BLINK-J** | 选项匹配 | 88.00 | 88.67 | 88.00 | **88.22 ± 0.39** | 本地 3 个 result 可核查 |
| **V\*** | 选项匹配 | 68.59 | 68.06 | 67.54 | **68.06 ± 0.53** | 本地 3 个 result 可核查 |
| **HRBench** | GPT-4o 选项提取器 | 63.12 | 63.12 | 62.88 | **63.04 ± 0.14（不可作为终值）** | 每 run 有 108–111/800 条因 API 不可达回退为 `Z`；确定性重提取明确 final-answer 格式后得到可审计下界 69.00/69.00/68.75，即 **≥68.92 ± 0.14**，仍需统一重判全部 `Z` 样本 |
| **GUIChat** | Qwen2.5-72B-Instruct judge | 73.70 | 73.49 | 73.60 | **73.60 ± 0.11** | 本地 3 个 `result_judged.jsonl` 可核查；962 样本/run |
| **WebMMU Functional (Act.)** | Qwen2.5-72B-Instruct judge | 72.15 | 71.14 | 71.95 | **71.75 ± 0.53** | 执行日志记录完成；当前工作副本只有 run1/2 各 111 条 checkpoint，无 run3 目录、full result 或 judged artifact，提交前必须同步验证 |

### 解释与限制

- 不能继续使用旧的 3B-judge GUIChat 80.59 或错误 `webquest` 数据上的 WebMMU 63.04；最终口径只保留论文同款 72B judge。
- WebMMU 的论文 `Act.` 对应正确 `webmmu` task 中的 `Functional` 子类；执行日志记录完整 English split 为 1,476 条、其中 Functional 492 条，但这些 full artifacts 尚未同步到当前工作副本。
- HRBench 当前 63.04 主要受评测器网络回退污染，不是可信模型终值。`≥68.92 ± 0.14` 仅是从现有输出中重提取明确答案得到的下界，不应替代统一重判后的正式分数。
- 三次重复仅刻画固定 checkpoint 的随机推理方差；不支持训练稳定性或统计显著性 claim。

## 二、成本—延迟拆分（正式 JSON 唯一口径）

| Benchmark | 主要工具类型 | generation | tool execution | orchestration | other / I/O | wall time |
|---|---|---:|---:|---:|---:|---:|
| BLINK-J | 本地算子 | 92.48% | **0.39%** | 0.01% | 7.11% | 175.9 s |
| Jigsaw | 本地算子 | 90.95% | **0.55%** | 0.02% | 8.49% | 1,275.8 s |
| GUIChat | OCR + Point + Crop | 73.85% | **18.65%** | 0.01% | 7.50% | 2,744.2 s |
| WebMMU fix（仅当前 111 条 checkpoint run） | OCR + Point + Crop | 87.10% | **10.17%** | <0.01% | 2.73% | 10,284.3 s |
| V\* | Point + Crop + OCR | 31.16% | **48.11%** | 0.05% | 20.69% | 1,411.9 s |
| VSPO | Point 专家模型密集 | 43.83% | **54.50%** | 0.01% | 1.66% | 13,427.2 s |
| VSP | Point 专家模型密集 | 39.11% | **59.42%** | 0.01% | 1.47% | 6,541.6 s |

硬件/服务口径：单 H20 上运行 7B 主模型（TP=1）；VSP/VSPO 使用 Point/Molmo 专家 worker。wall-clock 包含队列与 I/O，因此不应跨不同并发配置直接比较绝对时间。

微基准：AStar 本地算子 **0.092 ms/call**，Point/Molmo 专家模型 **255.333 ms/call**，约 **2,775×**。这支持“CPS 不是 wall-clock cost proxy”的窄结论，但不能替代 reviewer 要求的 matched-budget accuracy–latency 曲线。

## 三、E4 与 E5 审计结论

### E4：当前 rollout 不支持“选择性少用工具”

- VSP、Jigsaw 的工具使用率约 100%；难 VSP navigation 比 verify 使用更多调用，只能说明困难样本投入更多调用。
- GUIChat 三次 full rollout 均为 **962/962 样本使用至少一次工具，0 个 no-tool 样本**。
- 因而现有数据不能估计“正确且不调用工具”的比例，也不能证明 asymmetric reward 带来 cost-aware abstention。最终回复必须删除该实证 claim，或补 easy-task/no-tool 对照及 reward ablation。

### E5：准确率变化可信，detect/react 指标仅为启发式

- VSP/Jigsaw 各 100 条、五类 early fault 的逐条件结果已生成。
- 可直接报告 accuracy delta：VSP 最差为 timeout **−6 pp**；Jigsaw 最差为 missing **−17 pp**。
- 当前 `detect` 定义把 fault 后任意再次调用工具计作检测；clean baseline 也得到 detect=1.0，说明该指标没有 fault-specific 校准。`propagate` 同样依赖该启发式。
- 在人工抽检或 baseline-adjusted 重分析前，不应写“模型检测了 70–100% 故障”，也不能把恢复行为因果归因于 failure/reflection 训练轨迹。

## 四、Judge 审计（500 条）

- 证据包：`rebuttal_content/judge_audit_500_selected_20260804.tar.gz`。当前 selected archive 实际包含 `audit_500_detailed.jsonl`、`audit_decisions.json`、`disagreements.md`、`readme.md`、`report.md` 五个文件。
- `readme.md` 还引用了未打包的 sample CSV/Markdown、`metrics.json`、judge prompt、sampling manifest、checksums 和两个复现脚本；正式提交/开源前应补成完整归档。
- 抽样：GUIChat 197 条、WebMMU 303 条，按总体比例分层并在八个被评模型间近似均衡；固定随机种子 `260118631`。
- 有效样本 498/500；两条 WebMMU 记录只有通用任务前缀且 gold 为空，排除。
- Qwen2.5-72B judge 与逐条宽松语义复核的一致率为 **90.76%（452/498；Wilson 95% CI 87.90%–93.00%）**，**Cohen's κ=0.781**。
- Precision/recall/specificity 为 **95.59%/91.29%/89.44%**；FP=15，FN=31。
- 分 benchmark：GUIChat **86.29%（170/197）**，WebMMU **93.69%（282/301）**。
- 答案字符长度四分位的一致率为 87.10%/91.20%/93.55%/91.20%，没有对最长答案的单调一致率优势；该分析是描述性的，不等同于控制 correctness 后的因果回归。
- 重要限制：包内 reviewer 字段为 `Codex（逐条语义复核）`。这是一轮可复现的单 reviewer semantic audit，不应写成“两名作者人工盲标”。若正文使用 `human validation`，仍需作者确认或第二个独立标注者。

## 五、当前可提交结论与硬缺口

### 已有较强证据

1. 固定 checkpoint 下，多次随机推理的方差较小（仅 inference-repeat variance）。
2. 本地算子与专家模型工具的每调用成本相差约三数量级，CPS 会掩盖工具异质性。
3. Early fault injection 会造成任务相关的 accuracy drop，timeout/missing 是明显薄弱点。
4. GUIChat 使用论文同款 72B judge 后为 73.60 ± 0.11，协议已对齐。
5. 500 条 judge semantic audit 达到 90.76% agreement、κ=0.781，并提供完整 disagreement 理由。

### 提交前硬缺口（按优先级）

1. 同步 WebMMU 三次 full result/judged artifacts，并重算表中 71.75 ± 0.53。
2. 对 HRBench 所有 `Z` 回退样本进行统一、离线、可复现的答案重提取，冻结正式三次结果。
3. 完成 matched-budget accuracy–latency 曲线；当前只有阶段拆分和微基准。
4. 对 E5 做人工抽检/校准，补 late-turn injection；若声称训练归因，还需 with/without failure-reflection 训练消融。
5. 若要在论文中使用 `human validation`，由作者盲审确认现有 500 条中的子集，或增加第二个独立人类标注者；若只称 `single-reviewer semantic audit`，当前 agreement/κ 已可报告。控制 correctness 的正式长度回归仍是增强项。
6. 在完整 Jigsaw train/test source images 上完成 pHash+CLIP 近重复检查。
7. 核对 332,649 cold-start 样本分解，并把 manuscript、表图编号、引用和 point-by-point PDF 真正落地。

## 六、证据路径

- 多次结果：`rebuttal_exps/E_*/result*.jsonl`、`rebuttal_exps/E2_*/`、`rebuttal_exps/E3_*/`
- 阶段时延：`rebuttal_exps/*/latency.json`
- 工具微基准：`rebuttal_exps/E3_tool_latency.json`
- GUIChat 72B judge：`rebuttal_exps/E_guichat*/result_judged.jsonl`
- Judge audit：`.agent/ref/reviews/rebuttal_content/judge_audit_500_selected_20260804.tar.gz`
- Fault matrices：`rebuttal_exps/E5_matrix.json`、`rebuttal_exps/E5j_matrix.json`
- 执行记录：`.agent/exec_log.md`
