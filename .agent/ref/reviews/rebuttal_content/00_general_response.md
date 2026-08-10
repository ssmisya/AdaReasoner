# Response to Reviewers — General Response

We thank both reviewers for their careful, detailed, and constructive reviews. Their comments converge on four themes: (i) the relationship to our prior conference version and the novelty boundary; (ii) the scope of the generalization claims; (iii) statistical and reporting rigor; and (iv) the cost and failure modes of multi-turn tool use. We address every point in the accompanying responses and will ensure that the revised manuscript makes the same distinctions.

## Relationship to our conference version

We explicitly acknowledge and cite our conference paper, *AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning* (Mingyang Song, Haoyu Sun, Jiawei Gu, Linjie Li, Ranjay Krishna, and Yu Cheng; The Fourteenth International Conference on Learning Representations, 2026; OpenReview `nUGPEmQ2ut`). The conference version contributed the trajectory-curation pipeline, Tool-GRPO, the composite reward, the seven-tool suite, and the single-task results. **The journal extension adds:**

1. identifier randomization and description paraphrasing for interface-robust tool use;
2. a systematic study under randomized cold-start and RL;
3. a tool-planning evaluation on V\* and HRBench.

We do not re-claim the conference contributions as new. We also narrow “new task” to **cross-stage transfer** and “new tool” to **interface-level robustness**, because the final policy sees the tasks during Tool-GRPO and the renamed tools preserve functionality.

## Reliability and cost

We add fixed-checkpoint stochastic-inference repeats and explicitly distinguish them from training-seed variance. Locally audited three-run results include VSP **89.27 ± 0.88**, Jigsaw-COCO **88.27 ± 0.12**, and GUIChat under the paper-aligned 72B judge **73.60 ± 0.11**. WebMMU full judged artifacts and uniformly re-scored HRBench results remain submission gates and will not be presented as final until verified.

We also report direct wall-clock decomposition and per-tool latency. Tool execution accounts for **0.55%** on Jigsaw but **59.42%** on VSP; AStar and Point take **0.092** and **255.333 ms/call**, respectively. These results show that calls per sample are not a cost proxy. A matched-budget accuracy–latency comparison is still required before making an efficiency claim.

## Evidence boundaries

We remove or qualify claims that the current experiments do not support. In particular, all three GUIChat runs use tools on every sample, so they do not demonstrate cost-aware no-tool abstention. Likewise, our early-fault accuracy results are reportable, but the current detect/react heuristic requires manual calibration before it can support fault-detection claims. For the LM judge, a 500-item semantic audit now gives 90.76% agreement (452/498; Wilson 95% CI 87.90%–93.00%) and Cohen’s κ=0.781, with all 46 disagreements documented. Because the archived reviewer is Codex rather than two blinded authors, we describe this precisely as a single-reviewer semantic audit; author confirmation is still required if the manuscript claims human validation. Full-source near-duplicate screening and the cold-start sample decomposition also remain incomplete.

> Internal note: exact manuscript section/table/figure numbers should be inserted only after the revised manuscript is finalized. The complete conference citation is stored in `../CITATION_iclr2026.md`.
