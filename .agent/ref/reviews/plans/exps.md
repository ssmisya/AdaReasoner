# GenReasoner Rebuttal — 补充实验状态（2026-08-04）

> 状态：Ready=证据/回复闭环；Partial=有结果但仍缺关键项；Blocked=关键结论无有效证据；Scoped=收窄 claim，不再执行原设计。

## 状态总览

| ID | 对应 review | 状态 | 当前结果 | 下一步 |
|---|---|---|---|---|
| E1 | R1-2 / R2-2 | Scoped | 不执行“真新能力工具”实验 | 将 claim 限为 interface robustness + cross-stage transfer |
| E2 | R2-3 | Partial | 6 个 benchmark 有本地三次推理重复；GUIChat 72B=73.60±0.11 | 同步 WebMMU full/judged；统一重判 HRBench `Z`；明确非 training seeds |
| E3 | R1-6 / R2-5 | Partial | 阶段拆分 + per-tool 微基准完成 | 补 matched-budget accuracy–latency 曲线 |
| E4 | R1-5 | Blocked | GUIChat 三次均 962/962 使用工具；no-tool n=0 | 删除 selective abstention claim，或补 easy/no-tool + reward ablation |
| E5 | R1-6c / R2-4 | Partial | 五类 early fault accuracy 完成 | 校准 detect/react；可选 late fault 和训练消融 |
| E6 | R2-7 | Partial | 500 条 semantic audit：498 有效，agreement=90.76%，κ=0.781，FP/FN=15/31 | 如要称 human validation，补作者盲审或第二独立标注；长度分析目前仅为描述性四分位 |
| E7 | R1-3 / R1-4 | Ready | matched GPT-5+Tools 数字和 baseline caveat 已有 | 把协议表与 caption 真正写入 manuscript |
| E8 | R2-6 | Partial | source-image-disjoint 声明已有 | 用完整 train/test 源图运行 pHash+CLIP |

## E2 — 多次推理与评分可靠性

### 已核查

| Benchmark | mean ± sample std | 证据 |
|---|---:|---|
| VSP | 89.27 ± 0.88 | 三个本地 full result |
| VSPO | 78.64 ± 0.33 | 三个本地 full result |
| Jigsaw-COCO | 88.27 ± 0.12 | 三个本地 full result |
| BLINK-J | 88.22 ± 0.39 | 三个本地 result |
| V\* | 68.06 ± 0.53 | 三个本地 result |
| GUIChat | 73.60 ± 0.11 | 三个本地 Qwen2.5-72B-Instruct judged result |

这些 run 没有固化显式 `seed:`，只能称 fixed-checkpoint stochastic inference repeats。

### 未闭环

- WebMMU：执行日志记录 72.15/71.14/71.95，但本地只见 run1/2 各 111 条 checkpoint，无 run3/full/judged artifact。
- HRBench：外部 answer extractor 失败后每 run 108–111 条回退为 `Z`。当前 63.04 不可提交；明确 final-answer 重提取给出 ≥68.92±0.14 下界，仍需统一重判。

## E3 — 成本和时延

| Task | generation | tool execution | orchestration | other / I/O |
|---|---:|---:|---:|---:|
| Jigsaw | 90.95% | 0.55% | 0.02% | 8.49% |
| VSP | 39.11% | 59.42% | 0.01% | 1.47% |

微基准：AStar 0.092 ms/call；Point/Molmo 255.333 ms/call；约 2,775×。

剩余实验：在一致硬件/服务条件下限制最大轮数或工具调用预算，对 GenReasoner 和 baseline 生成 accuracy–latency 曲线。没有该曲线前，不声称 favorable efficiency trade-off。

## E4 — Reward 自查

- VSP verification 2.00 calls/sample，navigation 5.28 calls/sample。
- Jigsaw 近 100% 使用工具。
- GUIChat 三次均为 962/962 样本至少调用一次工具。

结论：现有 rollout 不能估计 correct-and-tool-free 子集，也不能证明 asymmetric reward 产生 cost-aware abstention。默认策略是删除该 claim；reward ablation 属额外训练实验。

## E5 — Tool failure

| Fault | VSP acc / Δ from 0.34 | Jigsaw acc / Δ from 0.90 |
|---|---:|---:|
| plausible-but-wrong | 0.39 / +0.05 | 0.77 / −0.13 |
| missing | 0.36 / +0.02 | 0.73 / −0.17 |
| malformed | 0.29 / −0.05 | 0.84 / −0.06 |
| timeout | 0.28 / −0.06 | 0.81 / −0.09 |
| contradictory | 0.30 / −0.04 | 0.82 / −0.08 |

可报告 accuracy delta。当前 detect/react 不是 fault-specific：clean baseline 也为 1.0。人工校准前不报告“70–100% 检测率”或“近零传播”。只完成 early injection；late injection 和训练消融未完成。

## E6 — Judge 验证

已完成 500 条分层 semantic audit：GUIChat 197 条，WebMMU 303 条；其中 498 条有效。Qwen2.5-72B judge 与复核标签的 agreement 为 **90.76%（452/498；Wilson 95% CI 87.90%–93.00%）**，**κ=0.781**，FP/FN=15/31；GUIChat/WebMMU agreement 分别为 **86.29%/93.69%**。46 条标签翻转均有题目级理由。

字符长度四分位 agreement 为 87.10%/91.20%/93.55%/91.20%，最长答案没有单调一致率优势，但这只是描述性检查，不是控制 correctness 的正式回归。

限制：归档的 reviewer 字段为 `Codex（逐条语义复核）`，因此只能称 single-reviewer semantic audit。若要在论文中称 `human validation`，仍需：

1. 由作者盲审复核现有样本子集，或增加第二个独立人类标注者；
2. 记录 annotator、blinding 和 adjudication；
3. 可选补控制 correctness 的 answer-length 回归。

证据：`rebuttal_content/judge_audit_500_selected_20260804.tar.gz`。

## E7 — Baseline 协议

- Main-table proprietary models：no-tool、single-turn，必须显式标注。
- GPT-5+Tools：VSP 55.64→71.36；Jigsaw 80.10→84.50。
- DeepEyes/Pixel-Reasoner：未适配 multi-tool interface；结果只能支持 unseen-interface brittleness。

## E8 — Jigsaw leakage

先用完整构造 manifest 验证 source-image disjoint，再对完整 train/test source images 做：

- pHash Hamming distance ≤5；
- CLIP cosine similarity ≥0.95；
- 报告真实 overlap 数、flagged pairs 和人工复核结果。

当前副本只有测试侧材料，不能填写“预期 0”。
