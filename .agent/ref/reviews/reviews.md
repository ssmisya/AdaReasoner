Reviewer 1

Review Comments

This paper presents GenReasoner, a framework for enabling Multimodal Large Language Models (MLLMs) to perform adaptive, multi-turn tool planning for complex visual reasoning tasks. This paper demonstrates strong engineering and extensive experimental validation, showing that tool-augmented reasoning can yield substantial performance gains. However, significant concerns remain regarding the novelty of methodological contributions, rigor of experimental design, fairness of baseline comparisons, and theoretical grounding of reward design. The generalization experiments suffer from confounding factors, and the comparison with closed-source models lacks sufficient transparency. Hence, my decision regarding this paper is to make a Major Revision.
1.The three claimed innovations have substantial precedents: high-quality trajectory data extends existing CoT/CoM methods (CogCoM, TACO); Tool-GRPO is a direct application of GRPO without task-specific algorithmic modifications; adaptive learning via randomization and paraphrasing is a well-established technique in domain adaptation and meta-learning literature.
2.In the "Generalize to New Tools" experiment (Section 3.3), training and evaluation use tools with identical functionality but different names/descriptions. Since the model has been exposed to trajectories with semantically equivalent functionality during training, it is unclear whether generalization stems from understanding abstract tool functions or merely learning more robust interface mappings. The current design cannot disentangle these interpretations.
3.Closed-source model evaluation: It is unclear whether GPT-5, Claude Sonnet 4, etc., were provided with the same tool set and system prompts. If not, the claim of "surpassing GPT-5" compares a tool-augmented model against a base model, which is potentially misleading.
4.DeepEyes/PixelReasoner comparison: These baselines were originally designed for single-tool/fixed-loop scenarios; evaluating them on a new tool set without adaptation may be unfair. Table 6 shows low CPS and success rates, which could stem from prompt incompatibility rather than inherent inferiority.
5.The paper designs an asymmetric reward structure that encourages brevity when correct and tool use when incorrect. However, no theoretical justification or convergence analysis is provided. This design may induce reward hacking: the model might output direct answers when correct (receiving full reward) and only use tools when uncertain, contradicting the goal of using tools as reasoning enhancers.
6.GenReasoner introduces multi-turn tool interaction (averaging 3.5–4.5 turns), significantly increasing inference latency and computational cost. While the paper reports Call Per Sample (CPS), it does not analyze: (a) the additional latency impact for real-world deployment; (b) sample complexity as the tool set scales; (c) error propagation risks in multi-turn interactions when intermediate tool calls fail.

Reviewer 2

Summary

GenReasoner is a family of multimodal tool-planning models: synthetic trajectory curation, a supervised Tool Cold Start stage, a Tool-GRPO stage, and an adaptive-learning mechanism that randomizes tool identifiers and paraphrases descriptions. Gains on structured tasks (VSP, Jigsaw) are large, and the paper shows the model adopting useful tools, suppressing irrelevant ones, and operating under renamed tool definitions.

The core idea is relevant. My central reservation is architectural: a tool-augmented model does not remove fallibility, it redistributes it across tool selection, argument generation, execution, observation interpretation, and synthesis, and every added turn is one more site where a local error can reach the answer. The evaluation mostly characterizes what happens when the architecture succeeds, and says little about when it fails, what it costs, and how far its generalization reaches. A prior concern is that most of what the paper presents as new is already published (Major 1). The points below are meant as paths to a stronger paper.

Major concerns

1. This is an extension of published work, and the boundary is not drawn.

The method (curation pipeline, Tool Cold Start, Tool-GRPO, reward design, the seven-tool suite) and the single-task results in Table 2 are already published as AdaReasoner (ICLR 2026), same authors, down to identical cells (7B TC+TG: VSP 97.64, Jigsaw 96.60, +38.66% average) and identical trajectory figures. The manuscript presents these as new ("we introduce GenReasoner, a new family of state-of-the-art models") without separating them from the published version. What is actually new is the identifier-randomization Adaptive Learning (Sec 2.4), the generalization study with Rnd TC + Rnd TG (Table 4), and the tool-planning comparison on V*/HRBench (Tables 5 to 6). Extending a conference paper is legitimate, but the manuscript should (i) cite the published version explicitly, (ii) state precisely what is new, and (iii) concentrate its claims and evaluation on that delta. As written, the headline contribution restates published results. (If the reference list already cites the ICLR version, this reduces to (ii) and (iii); I could not verify the full reference list from my copy.)

2. The new generalization claims exceed the evidence.

This is where the genuinely new material sits, so it carries the most weight. For new tasks, VSP and WebQA are withheld only from Tool Cold Start; Section 3.3 states all three tasks' data is used during Tool-GRPO, so the final policy does see them. That is transfer across training stages, not zero-shot generalization to a new task family. For new tools, the evaluation toolset changes identifiers "while preserving the same underlying tool functionalities," which tests robustness to a tool's textual interface, not use of a genuinely new capability. Please scope the claims to interface-level robustness, or add tools with functionalities and I/O absent from all training, and tasks excluded from every stage.

3. Statistical reliability, and numbers that do not reconcile across the merge.

Results are single runs, no seeds, no intervals, in exactly the setting (Tool-GRPO plus stochastic multi-turn inference) where variance is largest. Worse, the base-model numbers disagree between the inherited and new tables: untrained Qwen2.5-VL-7B on GUIChat is 59.46 in Table 2 but 68.09 in Tables 4 and 5; the 3B base differs on WebMMU (55.89 vs 54.47) and GUIChat (45.11 vs 46.26). The same untrained model on the same benchmark cannot give two numbers. This reads as the published tables and the new tables being run under different conditions and left unreconciled. Please reconcile, state the evaluation conditions per table, and add multi-seed uncertainty.

4. The architecture is not evaluated under tool failure.

The cold-start design deliberately includes tool-failure and fallback trajectories, yet nothing measures robustness when tools actually fail. The one data point, the A* distractor in Table 5 (verification drops 94.20 to 80.00 when an irrelevant tool is exposed), is uncontrolled interference, not fault injection, and already shows the failure surface is real. A controlled study (incorrect-but-plausible outputs, missing or malformed responses, timeouts, contradictory tools, errors injected early vs late) measuring whether the model detects, ignores, recovers from, or propagates each fault would turn error propagation from an edge case into a characterized property. Include the ablation with and without failure and reflection trajectories.

5. Inference cost is never measured (bears on the published core, still needed).

Turns, CPS, and tool-success rate are reported; latency, throughput, and compute or monetary cost are not. CPS is not a cost proxy when the toolset mixes local operations with expert-model calls: identical CPS can differ by an order of magnitude in wall-clock time. Without an accuracy vs latency (or compute) curve against baselines at a matched budget, one cannot separate a useful system from an expensive test-time-compute strategy. Report latency distributions, a generation, execution, and orchestration breakdown, and per-tool times.

6. Possible image-level leakage in Jigsaw-COCO (published core; verify).

Section C.1 builds training from three patch positions of each image and tests on the fourth patch of the same images. That holds out a patch position, not an image, so the model may have seen most of a test image's content in training. Please confirm and enforce disjointness at the source-image level (image-disjoint splits or COCO-val), with a near-duplicate check. Since this concerns already-published results, at minimum the manuscript should state the guarantee explicitly.

7. The LM-judge is under-validated, including on the new benchmarks.

V*, WebMMU, and GUIQA depend on Qwen2.5-VL-72B as judge. The human check on V* is reported only as "consistent," with no sample size, annotator count, agreement, or blinding. Since tool-augmented answers are more verbose than baselines, a lenient judge may reward length. Provide a quantitative agreement study and a check that verbosity is not advantaged, especially for the new V*/HRBench results.

Minor concerns

Moderate two claims. "Despite never being explicitly trained to do so" sits against an Adaptive Reward (Sec 2.3/A.4) built to regulate tool use and an Adaptive Learning method (Sec 2.4) built for generalization; distinguish not-supervised-at-instance-level from not-trained-at-all. And the "bottleneck shifted from scale to tool quality" claim holds on structured tasks but is contradicted by the paper's own general-task gains (about 4 to 7 points on V*/HRBench/WebMMU vs about 40 on VSP/Jigsaw) and its own "cannot fully offset" caveat; present it as a structured-task finding.

Naming. The body says GenReasoner; Figures 1 and 10 and the linked repository say AdaReasoner. In Figures 1 and 10, "AdaReasoner" is plotted beside DeepEyes and PixelReasoner, so a reader may misread the model as a third-party baseline. Use one name throughout, and confirm the released code and checkpoints correspond to this manuscript.

Reproducibility. The data statement covers only public source datasets, not the generated trajectories, exact splits, VSPO and Jigsaw-COCO construction, or seeds. Commit to releasing these, and explain how the 332,649 cold-start samples decompose, since the appendix does not obviously sum to that figure.

Limitations. A dedicated section is missing (latency and cost, dependence on hand-designed trajectories and external expert models, task-specific tooling). Dependence on tool quality is discussed, but only as an advantage; state it also as a fragility.

Smaller items. VSPO test grids disagree (A.2 lists 5x5/7x7/9x9; C.1 lists 3x3/5x5/7x7/9x9, and A.2 calls 3x3 "larger"). Task count is inconsistent (four tasks in the main text vs "three challenging tasks"). Some table cross-references look off (Table 13 where 11 or 12 seem intended). Terminology varies (Tool-GRPO vs Tool GRPO). Distinguish a syntactically successful tool call from a semantically useful result, since a call can execute and still return misleading information.

Closing

The contribution is real: multi-turn, adaptive visual tool use with strong structured-task results. But most of it is already published, and the new material, the randomization-based generalization, is exactly where the evidence is thinnest and the tables least consistent. The paper would be substantially stronger if it drew a clear boundary against the ICLR version and concentrated its evidence on the delta, treating the failure structure and cost of the architecture as first-class rather than boundary cases.

