# GenReasoner Manuscript Revision — Change List

This document maps the journal reviews to the manuscript edits in
`.agent/ref/69a66abdb76ba160fb253194`.
All visible manuscript changes are marked in red in the paper.

## Main-text revisions

1. **Abstract**
   - Cite and distinguish the ICLR 2026 AdaReasoner conference paper.
   - Define the journal delta as interface-randomized Adaptive Learning,
     the randomized TC/TG study, and broader tool-planning evaluation.
   - Replace unrestricted “new tools/new tasks” language with
     interface robustness, cross-stage transfer, and benchmark-level transfer.
   - Scope proprietary-model comparisons to structured tasks and state the
     additional test-time cost/failure surface.

2. **Introduction and contributions**
   - Remove claims that trajectory curation, GRPO, and the seven-tool suite are
     new journal contributions.
   - Re-center the paper on the journal-specific empirical question:
     whether a published multi-tool framework remains robust when tool
     interfaces and task distributions change.
   - Replace “general-purpose tool skill” and unrestricted zero-shot claims
     with evidence-aligned formulations.

3. **Related work**
   - State that cold-start trajectory synthesis, GRPO, and
     randomization/paraphrasing have clear precedents.
   - Position Tool-GRPO as a multi-turn instantiation of GRPO rather than a new
     optimizer.
   - Add the fairness caveat that DeepEyes and PixelReasoner are evaluated
     without adaptation to the journal paper's multi-tool interface.

4. **Method**
   - Mark trajectory curation, Tool Cold Start, Tool-GRPO, reward design, and
     the tool suite as inherited from AdaReasoner.
   - Identify Adaptive Learning as the journal extension.
   - Scope identifier randomization and paraphrasing to interface robustness.
   - Remove the unsupported interpretation that the asymmetric reward produces
     cost-aware no-tool abstention.

5. **Experiments**
   - Label the single-task TC/TG results as inherited context.
   - Clarify that the “new-task” study is cross-stage transfer because all
     three tasks enter Tool-GRPO.
   - Clarify that renamed/rephrased tools preserve functionality and therefore
     test interface robustness, not mastery of genuinely new capabilities.
   - Distinguish no-tool, single-turn proprietary baselines from matched
     GPT-5+Tools conditions.
   - Add caveats for unadapted DeepEyes/PixelReasoner comparisons and define
     tool “success” as syntactic execution rather than semantic correctness.
   - Point readers to appendix-only reliability, latency, failure, and judge
     analyses.

6. **Discussion and limitations**
   - Add a dedicated limitations section covering latency/test-time compute,
     expert-worker dependence, hand-designed trajectories, interface-level
     rather than unrestricted generalization, and propagation of faulty tool
     observations.

7. **Conclusion and release statements**
   - Restate the conference/journal boundary.
   - Scope conclusions to the evaluated settings.
   - Expand the release commitment to generated trajectories, split manifests,
     construction scripts, prompts/rewards, checkpoint identifiers, and run
     metadata.

## Appendix revisions

1. **Jigsaw construction**
   - State explicitly that source images are sampled once and partitioned
     before puzzle construction, so patch position is not used as the
     train/test split key.

2. **Evaluation protocol**
   - Document which baselines receive tools and which are no-tool,
     single-turn baselines.
   - Correct the judge description to Qwen2.5-72B-Instruct.
   - Replace the unsupported “human evaluation is consistent” statement with
     the actual 500-item single-reviewer semantic audit.

3. **Revision analyses**
   - Add fixed-checkpoint stochastic-inference repeatability results.
   - Add wall-clock stage decomposition and per-tool latency.
   - Add controlled early-turn fault-injection results using auditable local
     artifacts.
   - Add the 500-item judge audit and response-length quartile check.
   - State explicitly what these analyses do and do not establish.

## Evidence deliberately not promoted to the manuscript

- No matched-budget accuracy–latency curve is currently complete.
- The newer 200-item VSP failure table is not used until its raw artifacts and
  exact protocol are synchronized; the appendix uses the locally auditable
  100-item VSP-navigation and 100-item Jigsaw experiments.
- HRBench repeatability is not reported until every `Z` fallback is uniformly
  re-scored offline.
- No pHash/CLIP near-duplicate result is claimed before the complete source
  image sets are available.
- The 500-item judge review is not called a two-human or author-blinded audit.
