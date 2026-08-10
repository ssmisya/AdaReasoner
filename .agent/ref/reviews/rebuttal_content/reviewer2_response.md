# Response to Reviewer 2

We thank the reviewer for the exceptionally detailed and constructive review, especially for identifying the boundary with the conference paper and the cross-table inconsistencies. Final manuscript section/table numbers will be inserted after the revision is typeset.

## R2-1 — Boundary with the published conference version

We agree and explicitly cite our conference paper, *AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning* (Mingyang Song, Haoyu Sun, Jiawei Gu, Linjie Li, Ranjay Krishna, and Yu Cheng; ICLR 2026; OpenReview `nUGPEmQ2ut`). The conference version contains trajectory curation, Tool-GRPO, the reward, the seven-tool suite, and the single-task results. The journal-only delta is: (i) identifier-randomization and description-paraphrasing Adaptive Learning; (ii) the randomized cold-start/RL generalization study; and (iii) the V\*/HRBench tool-planning evaluation. We will center the Introduction, contributions, and evaluation on this delta and present inherited results only as context.

## R2-2 — Generalization claims exceed the evidence

We agree and scope both claims:

- **Tasks:** this is cross-stage transfer, not zero-shot task-family generalization, because the final policy sees the tasks during Tool-GRPO.
- **Tools:** this is interface-level robustness to changed names, descriptions, schemas, or ordering while functionality is preserved, not mastery of a genuinely novel capability.

These narrower formulations will be used consistently in the Abstract, Introduction, experiment headings, table captions, and Conclusion.

## R2-3 — Statistical reliability and inconsistent values

We traced the inherited discrepancies to different prompts, judges, task loaders, and tool configurations. The revision will report one value per model–benchmark pair and bind it to an explicit configuration and result file.

The current fixed-checkpoint, stochastic-inference repeats are:

| Benchmark | run 1 | run 2 | run 3 | mean ± sample std | audit status |
|---|---:|---:|---:|---:|---|
| VSP | 89.91 | 89.64 | 88.27 | **89.27 ± 0.88** | three local full results |
| VSPO | 78.98 | 78.32 | 78.62 | **78.64 ± 0.33** | three local full results |
| Jigsaw-COCO | 88.20 | 88.20 | 88.40 | **88.27 ± 0.12** | three local full results |
| BLINK-J | 88.00 | 88.67 | 88.00 | **88.22 ± 0.39** | three local results |
| V\* | 68.59 | 68.06 | 67.54 | **68.06 ± 0.53** | three local results |
| GUIChat, Qwen2.5-72B-Instruct judge | 73.70 | 73.49 | 73.60 | **73.60 ± 0.11** | three local judged results |

We explicitly scope these as **inference-repeat variance for a fixed checkpoint**, not training-seed variance or a significance test.

Two rows are not yet submission ready:

1. **WebMMU Functional/Act.** The execution log records 72.15/71.14/71.95 (**71.75 ± 0.53**) under the corrected task and 72B judge, but the current working copy contains only 111-item checkpoints for runs 1–2 and no run-3/full/judged artifacts. We will synchronize and recompute these files before using the number.
2. **HRBench.** The current 63.12/63.12/62.88 scores contain 108–111 of 800 items per run mapped to `Z` after the external answer-extraction API became unavailable. Deterministic recovery of only explicit final-answer formats gives auditable lower bounds of 69.00/69.00/68.75 (**≥68.92 ± 0.14**), but the final row requires uniform offline re-scoring of every `Z` item.

## R2-4 — Tool failure

We ran five **early-turn** fault conditions on fixed 100-item subsets. Because the current detect/react heuristic is not fault specific, we report the directly auditable task accuracy first:

| Fault | VSP accuracy (baseline 0.34) | Δ | Jigsaw accuracy (baseline 0.90) | Δ |
|---|---:|---:|---:|---:|
| plausible-but-wrong | 0.39 | +0.05 | 0.77 | −0.13 |
| missing | 0.36 | +0.02 | 0.73 | **−0.17** |
| malformed | 0.29 | −0.05 | 0.84 | −0.06 |
| timeout | 0.28 | **−0.06** | 0.81 | −0.09 |
| contradictory | 0.30 | −0.04 | 0.82 | −0.08 |

These results show task-dependent sensitivity, with timeout the largest VSP degradation and missing responses the largest Jigsaw degradation. We do **not** currently claim a 70–100% detection rate: the heuristic treats any subsequent tool call as detection and also returns 1.0 on the clean baseline. We will either manually calibrate it or omit detect/propagate rates. Late-turn injection and the with/without failure-reflection training ablation remain uncompleted, so we do not causally attribute robustness to those trajectories.

## R2-5 — Inference cost

The audited wall-time breakdown is Jigsaw generation/tool execution = **90.95%/0.55%** and VSP = **39.11%/59.42%**. The local AStar operator takes **0.092 ms/call**, while Point/Molmo takes **255.333 ms/call**, a **2,775×** difference. Thus CPS obscures tool-cost heterogeneity. These measurements do not replace the requested efficiency comparison: a matched-turn/tool-budget accuracy–latency curve against baselines remains a submission gate.

## R2-6 — Possible Jigsaw image leakage

The intended construction is source-image disjoint: train and test puzzles use different COCO source images, not different patch positions from the same source image. This statement must be verified against the construction manifests. The additional pHash (Hamming ≤5) and CLIP-cosine (≥0.95) near-duplicate screen has **not** been run in the current working copy because the full train source images are absent. We will report the true overlap and flagged-pair counts only after running it on the complete source sets.

## R2-7 — LM-judge validation

We completed a 500-item stratified semantic audit of Qwen2.5-72B-Instruct decisions over GUIChat and WebMMU, approximately balanced across eight evaluated models. Of 500 sampled records, 498 are valid and two empty-reference WebMMU records are excluded. Agreement is **90.76% (452/498; Wilson 95% CI 87.90%–93.00%)**, with **Cohen’s κ=0.781**, precision/recall/specificity of **95.59%/91.29%/89.44%**, 15 false positives, and 31 false negatives. Agreement is **86.29% (170/197)** on GUIChat and **93.69% (282/301)** on WebMMU. All 46 disagreements have item-level reasons in the archived audit package.

A descriptive character-length analysis gives agreement rates of 87.10%, 91.20%, 93.55%, and 91.20% across increasing length quartiles, providing no monotonic agreement advantage for the longest answers; we do not treat this descriptive result as a causal verbosity test. Importantly, the archive identifies the semantic reviewer as `Codex（逐条语义复核）`. We therefore describe this as a large, reproducible **single-reviewer semantic audit**, not as a two-author human study. If the manuscript retains the term “human validation,” an author-blinded confirmation or independent second annotation pass remains necessary.

## Minor points

- We will replace “never explicitly trained” with the narrower “not supervised at the instance level.”
- We will scope “the bottleneck shifts from scale to tool quality” to the studied structured tasks.
- We will use one model name consistently and verify figures, repository labels, and checkpoints.
- We will release trajectories, exact splits, construction scripts, and inference-run metadata; the three repeats do not have explicit training seeds.
- The **332,649** cold-start samples still require a verified task/stage decomposition that sums to the reported total.
- We will add limitations covering latency, expert-worker dependence, task-specific tools, hand-designed trajectories, and tool-quality fragility.

> **Internal submission gates:** synchronize WebMMU full/judged artifacts; uniformly re-score HRBench `Z` items; calibrate E5 or report accuracy only; run pHash+CLIP; decide whether to report the completed single-reviewer audit as such or add author/second-human confirmation; verify the 332,649 decomposition.
