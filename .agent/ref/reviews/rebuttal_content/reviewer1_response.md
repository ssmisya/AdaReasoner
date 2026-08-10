# Response to Reviewer 1

We thank the reviewer for the detailed assessment and for recognizing the engineering strength and extensive validation of the work. We address each point below. Final manuscript section/table numbers will be inserted after the revision is typeset.

## R1-1 — Novelty of the methodological contributions

We agree that SFT cold-start, GRPO, and interface randomization each have clear precedents, and we will remove wording that implies these components are individually novel. We explicitly distinguish the inherited core from our conference paper, *AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning* (ICLR 2026), from the journal-only additions. The inherited core comprises trajectory curation, Tool-GRPO, the composite reward, the seven-tool suite, and the single-task results. The journal delta is: (i) identifier randomization and description paraphrasing for interface robustness; (ii) the randomized cold-start/RL generalization study; and (iii) the V\*/HRBench tool-planning evaluation. We therefore position the contribution as system integration and new empirical analysis, not a new RL algorithm.

## R1-2 — “Generalize to New Tools” tests interface remapping

We agree and scope the claim. We no longer claim abstract tool-function understanding or zero-shot mastery of a genuinely novel capability. The experiment supports **interface-level robustness**: functionality is preserved while tool identifiers, descriptions, and ordering change. The task result is likewise **cross-stage transfer**, because the final policy sees the tasks during Tool-GRPO even when they were withheld from Tool Cold Start. We retain tool-selection evidence only as evidence of routing under interface changes, not capability acquisition.

## R1-3 — Transparency of closed-source model evaluation

We will add a protocol table specifying, for every baseline, whether it receives the same tools, system prompt, and multi-turn protocol. In the main inherited tables, GPT-5-20250807, Claude-Sonnet-4, and Gemini-2.5-Flash are no-tool, single-turn baselines; the manuscript will label that comparison explicitly rather than presenting it as protocol matched. We also report the existing matched-protocol GPT-5+Tools condition: VSP improves from 55.64 to 71.36 and Jigsaw from 80.10 to 84.50. We scope the claim to structured visual-reasoning tasks and acknowledge that proprietary models remain stronger on some open-ended tasks.

## R1-4 — Fairness of DeepEyes / Pixel-Reasoner

We agree that DeepEyes and Pixel-Reasoner were designed for single-tool or fixed-loop settings. They were evaluated without adaptation to our multi-tool interface. Their low engagement therefore reflects interface and prompt mismatch at least in part and cannot establish inherent inferiority. We will add this caveat to the table caption and describe the result only as brittleness under an unseen interface. We do not report nonexistent post-adaptation results.

## R1-5 — Asymmetric reward and possible reward hacking

We agree that our current evidence does **not** establish cost-aware abstention. A full-result audit found:

- VSP verification uses 2.00 tool calls per sample, while the harder navigation subset uses 5.28 calls; this is a difficulty correlation, not causal evidence about the reward.
- Jigsaw uses tools on approximately all samples.
- Across all three GUIChat runs, **962/962 samples per run use at least one tool**; the no-tool subset has size zero.

Consequently, we cannot report a “correct-and-tool-free” subset or claim that the asymmetric reward selectively suppresses unnecessary calls. We will remove that empirical claim unless we add an easy/no-tool control and a symmetric-versus-asymmetric reward ablation. We will describe the composite signal only as bounded reward shaping within the standard GRPO objective and explicitly state that we provide neither a separate convergence proof nor a causal reward ablation.

## R1-6 — Multi-turn cost, scaling, and error propagation

### (a) Latency and cost

We now measure wall-clock components directly rather than treating calls per sample as a cost proxy. Under the audited full-result files on a single H20 (GenReasoner-7B, TP=1):

| Task | generation | tool execution | orchestration | other / I/O |
|---|---:|---:|---:|---:|
| Jigsaw | 90.95% | **0.55%** | 0.02% | 8.49% |
| VSP | 39.11% | **59.42%** | 0.01% | 1.47% |

A micro-benchmark gives **0.092 ms/call** for the local AStar operator and **255.333 ms/call** for the Point/Molmo expert worker, a **2,775×** difference. This supports the narrow conclusion that CPS is not a wall-clock proxy. It does not yet answer the requested test-time-compute trade-off. A matched-budget accuracy–latency curve remains required before submission; we will not claim that existing stage totals constitute that curve.

### (b) Scaling with the exposed tool set

The existing enlarged-tool-set analysis suggests calls remain concentrated on relevant tools, but it is inherited evidence rather than a new controlled scaling experiment. We will report the exact exposed set and call distribution and avoid claiming general sample-complexity scaling from one setting.

### (c) Failed calls

We ran five early-fault conditions on fixed 100-item VSP and Jigsaw subsets. The directly auditable evidence is the accuracy change: the largest VSP drop is **6 percentage points** under timeout; the largest Jigsaw drop is **17 points** under a missing response. However, the current `detect/react` heuristic counts any post-fault tool call as detection and also equals 1.0 on the clean baseline. We therefore do not claim 70–100% fault detection or near-zero propagation without manual calibration. Late-turn injection and the with/without failure-reflection training ablation remain uncompleted.

> **Internal submission gates:** add the matched-budget accuracy–latency figure; calibrate the E5 detection metric or report only accuracy deltas; ensure all manuscript claims match these evidence boundaries.
