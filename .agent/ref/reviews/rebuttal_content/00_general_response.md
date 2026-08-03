# Response to Reviewers — General Response

> 说明(给作者,不进提交稿):`〖待补: ...〗` = 需要你们跑完对应实验/核对后填真实数字,**不要留编造值**。本目录按 point 拆开:`reviewer1_response.md`、`reviewer2_response.md`。

We thank both reviewers for their careful, detailed, and constructive reviews. The comments converge on four themes: (i) the relationship to our prior conference version and the novelty boundary; (ii) the scope of our generalization claims; (iii) statistical and reporting rigor; and (iv) characterizing the architecture's cost and failure modes. We have revised the manuscript substantially to address every point. We summarize the most important change here and give point-by-point responses in the accompanying letters.

## Relationship to our conference version (addresses R1-1 and R2-1)

We now explicitly acknowledge and cite our conference paper, **AdaReasoner (ICLR 2026)** [CITE], at first mention. The conference version contributed the trajectory-curation pipeline, the Tool-GRPO algorithm, the composite reward, the seven-tool suite, and the single-task results reported in Table 2. **This journal extension makes three new contributions beyond the conference version:**

1. an **identifier-randomization and description-paraphrasing Adaptive Learning** method for interface-robust tool use (Sec 2.4);
2. a **systematic generalization study** under randomized cold-start and RL (Rnd TC + Rnd TG, Table 4);
3. a **tool-planning evaluation on V\* and HRBench** (Tables 5-6).

We have rewritten the Introduction and the contribution list to (i) cite the conference version, (ii) state precisely what is new, and (iii) re-center the paper's claims and evaluation on this delta. Inherited results are now presented only as background/context, not as new headline claims.

**Revised contribution statement (manuscript, Sec 1):**
> "This work extends our conference paper (AdaReasoner, ICLR 2026 [CITE]). The conference version established the trajectory-curation pipeline, the Tool-GRPO algorithm, the reward design, and the seven-tool suite, achieving single-task state-of-the-art results (Table 2). Building on it, this article contributes: (i) an identifier-randomization / description-paraphrasing Adaptive Learning method that yields interface-robust tool use (Sec 2.4); (ii) a systematic generalization study under randomized cold-start and RL (Table 4); and (iii) a tool-planning evaluation on V\* and HRBench (Tables 5-6). We do not re-claim the conference results as new."
