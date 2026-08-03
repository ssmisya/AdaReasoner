# Point-by-Point Response to Reviewers

> Consolidated rebuttal for Springer Major Revision. Assembled from the General Response and the two per-reviewer letters, with the results of the revision experiments (E2–E8) integrated. Items still requiring authors' real data are marked **[AUTHORS TO FILL]** and listed at the end; these must be completed with genuine measurements before submission (do not fabricate).

We thank both reviewers for their careful, detailed, and constructive reviews. The comments converge on four themes: (i) the relationship to our prior conference version and the novelty boundary; (ii) the scope of the generalization claims; (iii) statistical and reporting rigor; and (iv) characterizing the architecture's cost and failure modes. We have revised the manuscript substantially to address every point. Manuscript locations are given as (Sec/Table X).

---

## General Response — Relationship to our conference version (addresses R1-1 and R2-1)

We now explicitly acknowledge and cite our conference paper, **AdaReasoner (ICLR 2026)** [CITE], at first mention. The conference version contributed the trajectory-curation pipeline, the Tool-GRPO algorithm, the composite reward, the seven-tool suite, and the single-task results in Table 2. **This journal extension makes three new contributions beyond the conference version:**

1. an **identifier-randomization and description-paraphrasing Adaptive Learning** method for interface-robust tool use (Sec 2.4);
2. a **systematic generalization study** under randomized cold-start and RL (Rnd TC + Rnd TG, Table 4);
3. a **tool-planning evaluation on V\* and HRBench** (Tables 5-6).

**Revised contribution statement (manuscript, Sec 1):**
> "This work extends our conference paper (AdaReasoner, ICLR 2026 [CITE]). The conference version established the trajectory-curation pipeline, the Tool-GRPO algorithm, the reward design, and the seven-tool suite, achieving single-task state-of-the-art results (Table 2). Building on it, this article contributes: (i) an identifier-randomization / description-paraphrasing Adaptive Learning method that yields interface-robust tool use (Sec 2.4); (ii) a systematic generalization study under randomized cold-start and RL (Table 4); and (iii) a tool-planning evaluation on V\* and HRBench (Tables 5-6). We do not re-claim the conference results as new."

Inherited results are now presented only as background/context, and the paper is renamed consistently to **GenReasoner** throughout (see Minor points).

---

# Response to Reviewer 1

We thank the reviewer for recognizing the engineering strength and extensive validation of the work.

## R1-1 — Novelty of the methodological contributions
We agree that "SFT cold-start + RL", GRPO, and interface randomization each have clear precedents, and we have removed any wording implying these components are individually novel. We now explicitly cite CogCoM and TACO for trajectory/CoM-style data, note that Tool-GRPO applies GRPO without a new RL operator, and cite prior domain-adaptation / meta-learning work for randomization/paraphrasing. We reposition our contribution as (a) the **integration** of these ingredients into a multi-turn, multi-tool orchestration framework, and (b) the **new material of this journal version** — interface-robustness Adaptive Learning (Sec 2.4), the randomized generalization study (Table 4), and the V\*/HRBench evaluation (Tables 5-6). See revised Sec 1 and Related Work.

## R1-2 — "Generalize to New Tools" cannot disentangle abstract-function understanding from robust interface mapping
We agree and have **scoped the claim**. We no longer claim the model understands "abstract tool functions"; we claim only **robustness to interface-level variation** (renamed / re-described / re-ordered tools with functionality preserved), together with cross-stage transfer, and state this explicitly in Sec 3.3 and the Table caption. This remains deployment-relevant: tool names, argument schemas, and descriptions change across versions and providers, and brittle interface-coupling is a real failure mode. As supporting evidence we retain the tool-selection analysis (the model adopts useful tools, ignores irrelevant ones, and down-weights redundant ones), which speaks to sensible tool selection under interface change rather than mastery of genuinely novel capabilities.

## R1-3 — Transparency of closed-source model evaluation
We have clarified the protocol (Sec 4.1 and Appendix) and added a table specifying, **for every baseline, whether it received the same tool set, the same system prompt, and the same multi-step protocol**. To be transparent: in the main tables the proprietary models (GPT-5-20250807, Claude-Sonnet-4, Gemini-2.5-flash) were evaluated **without tools, single-turn** (temperature 0, max 2048 tokens, no ICL, one question round) — a **tool-augmented open model vs. no-tool proprietary model** comparison, now labeled as such. We additionally report a **matched-protocol** comparison in which GPT-5 (and Qwen2.5-VL-7B/72B) run **inside our tool-server framework with the same tools and multi-step protocol** ("+Tools"): GPT-5+Tools improves markedly (VSP 55.64 → 71.36, Jigsaw 80.10 → 84.50), yet our TC+TG 7B model still leads on the structured visual-reasoning tasks (VSP 97.64, Jigsaw 94.20). We soften the phrasing accordingly: GenReasoner surpasses GPT-5 on structured visual-reasoning tasks — including when GPT-5 is given the same tools — while the proprietary models retain an edge on open-ended general tasks such as WebMMU.

## R1-4 — Fairness of the DeepEyes / Pixel-Reasoner comparison
We agree these baselines were designed for single-tool / fixed-loop settings. We **explicitly note that DeepEyes and Pixel-Reasoner were run without adaptation** to our multi-tool interface, and that their low CPS / success in Table 6 partly reflects **prompt and tool-interface incompatibility** rather than inherent inferiority: under our zero-shot tool-definition shift these methods show limited tool engagement and low execution reliability, which is exactly the brittleness to interface change our method is designed to avoid. We revise the text so the comparison reads as "our framework is more robust to unseen tool interfaces," **not** as a claim of inherent superiority, and add this caveat to the Table 6 caption. We refrain from reporting fabricated post-adaptation numbers; a fair minimal-adaptation re-evaluation is noted as a concrete follow-up.

## R1-5 — Reward design: theoretical justification and possible reward hacking
The asymmetric reward is a **deliberate design intention** that the manuscript stated unclearly. Tool calls incur latency and compute cost (see R1-6a: an expert-model tool call is ~2,800× more expensive than a local operator, and generation dominates end-to-end latency), so the reward is designed to elicit tool use **only when it improves correctness** — cost-aware, need-based tool use. "Solving without a tool when the model already can" is the *intended* cost-saving behaviour, not a failure mode. We now state this in Sec 2.3 and support it with a self-check on the full test sets:

- **Tool use scales with instance difficulty (need-based investment).** On VSP (full tool set incl. the expert-model Point tool), the easier *verification* sub-task uses **2.00 calls/sample** at **0.968** accuracy, whereas the harder *navigation* sub-task uses **5.28 calls/sample** at 0.368 — ~2.6× more calls where the task is harder. By difficulty level, average calls rise monotonically (L1: 2.0 calls, acc 1.00 → L6: 5.26, acc 0.28 → L8: 5.42, acc 0.29). On Jigsaw the model uses tools on ~100% of instances at **3.08 calls/sample** for 0.882 accuracy. In every case the model invests more calls precisely where the task demands it — the opposite of "avoid tools when you will need them."
- **Adopt-vs-discard behaviour (Sec 4.4).** During RL the model *increases* A\* usage on navigation (where it helps) while driving A\* usage toward *zero* on verification (where A\* is a distractor), keeping near-perfect (99.20) verification accuracy.
- **On "solving without tools has no accuracy penalty":** this cannot be measured on VSP/Jigsaw because both are perception/planning-hard and the model calls tools on ~100% of instances; we therefore make this specific claim on a general task where no-tool solutions are common (GUIChat), reporting the fraction of correct-and-tool-free instances and its accuracy. **[AUTHORS TO FILL: GUIChat no-tool self-check — % of correct answers produced without any tool call, and that subset's accuracy.]**

On theory: we optimize the standard GRPO objective (its convergence properties are inherited); our composite reward is **bounded reward shaping** that biases *when* tools are used and does not change the task optimum. We note the absence of a formal convergence proof as a scoped, honest limitation.

## R1-6 — Multi-turn cost, sample complexity, and error propagation
**(a) Latency / cost.** We now report inference cost directly (Sec 4 / Appendix): latency distributions, a per-stage breakdown (generation / tool-execution / orchestration), per-tool times, and an accuracy-vs-latency curve at a matched tool budget. Measured on **full test sets (GenReasoner-7B, single H20, TP=1)**, the tool-execution share depends on the **type** of tool:

| Task (tool set) | generation | tool-execution | orchestration |
|---|---|---|---|
| Jigsaw (local-only: DetectBlackArea, InsertImage) | 91.0% | **0.55%** | 0.02% |
| VSP (incl. expert-model Point / Molmo-7B) | 50.8% | **48.1%** | 0.007% |

A per-tool micro-benchmark explains the gap: a **local operator (A\*) costs ~0.09 ms/call** versus an **expert-model call (Point / Molmo-7B) ~255 ms/call — a ~2,800× difference**. This is precisely why **CPS is not a wall-clock proxy**: it counts a 0.09 ms local op and a 255 ms expert-model call as one equal "call", so in a mixed tool set CPS is dominated by cheap ops while *time* is dominated by the few expert-model calls. It also reframes the cost concern: the real cost of tool augmentation comes from expert-model calls, so cost-aware, need-based use of expensive tools — what the adaptive reward encourages — is exactly where latency is saved (R1-5). (The VSP figure uses a single Point worker, so tool-execution includes queueing and is an upper bound; parallel expert-model serving lowers wall-clock but not per-call cost.) **[AUTHORS TO FILL: accuracy-vs-latency curve figure vs. baselines at matched budget.]**

**(b) Sample complexity as the tool set scales.** When we expose the model at inference to an enlarged tool set (adding held-out perceptual tools GetStartPoint / GetEndPoint / GetObstacles and manipulation tools RotateImage / DrawDashLinePath on top of A\* / Point / Draw2DPath), the call distribution stays sharply concentrated: on VSP the model calls the capability-complementing A\* frequently (avg 0.77 calls/sample, 96.6% success) and Point / Draw2DPath heavily (2.11 / 0.61 calls/sample), while irrelevant tools receive near-zero calls (RotateImage 0.00, GetObstacles 0.00) and tools redundant with an already-mastered one are down-weighted (GetStartPoint 0.01). Total calls do **not** scale with tool-set size — the model routes to the few relevant tools.

**(c) Error propagation under failed calls.** We add a controlled tool-failure study (R2-4 / new Table X) injecting five fault types into tool responses and measuring detection/recovery/propagation. The model **detects and reacts to the large majority of injected faults** (VSP 0.70–1.00, Jigsaw 0.92–1.00) with **near-zero propagation** except VSP-timeout (0.24). Accuracy under early faults degrades by at most a few points on the multi-turn task (VSP, worst −6) and more on the single-shot task (Jigsaw, up to −17), reflecting different recovery headroom. This characterizes error propagation quantitatively rather than leaving it as an untested edge case.

---

# Response to Reviewer 2

We thank the reviewer for an exceptionally detailed review, and in particular for identifying the boundary-with-prior-work issue and the cross-table inconsistencies.

## R2-1 — This is an extension of published work; the boundary is not drawn
We fully agree and have redrawn the boundary: (i) we cite AdaReasoner (ICLR 2026) [CITE] at first mention; (ii) we state precisely what is new (Sec 2.4 Adaptive Learning; Table 4 Rnd TC + Rnd TG; Tables 5-6 V\*/HRBench); and (iii) we re-center the claims and evaluation on this delta, presenting inherited results only as background. See the General Response and the revised Introduction.

## R2-2 — The new generalization claims exceed the evidence
We agree and have scoped both claims.
- **New tasks.** We describe the setting precisely as **cross-stage transfer**: only Tool Cold Start withholds VSP/WebQA, whereas Tool-GRPO uses all three tasks, so the final policy does see them. We no longer describe this as zero-shot generalization to a new task family (Sec 3.3 revised).
- **New tools.** We scope this to **interface-level robustness** (identifiers/descriptions changed while functionality is preserved) and explicitly state we do **not** claim use of a genuinely novel capability. We considered constructing tools with entirely new functionality held out from all stages; we note in the text why a fair such test is hard to design (tool relevance is inherently task-coupled), and therefore make the narrower, well-supported claim.

## R2-3 — Statistical reliability, and numbers that do not reconcile
**Reconciliation.** The base-model discrepancies (Qwen2.5-VL-7B GUIChat 59.46 in Table 2 vs 68.09 in Tables 4-5; 3B GUIChat 45.11 vs 46.26; 3B WebMMU 55.89 vs 54.47) arose because the single-task table (Table 2) and the later generalization/main tables were produced under **different evaluation configurations** (an earlier GUIChat prompt/judging setup for Table 2 vs. the unified protocol for the generalization and final-model tables). We have **re-run the affected settings under a single unified protocol**, now report **one number per (model, benchmark)** (e.g. 68.09 for 7B GUIChat, consistent across the generalization, randomization, and final-model tables), and state the evaluation conditions per table so no stale value remains.

**Variance.** All key results are now reported over **multiple inference seeds as mean±std**, with two complementary pieces of evidence:
1. **A 3-seed variance table** (fixed checkpoint, 3 independent runs) on the main VSP/Jigsaw configurations shows within-configuration variance is small relative to between-configuration gaps: VSP Overall Qwen2.5-VL-7B 28.98 (±1.13) → +TG 73.34 (±0.11) → +TC+TG 97.02 (±1.35); Jigsaw-COCO 44.00 (±1.80) → 81.22 (±5.95) → 95.98 (±1.19). The tens-of-points improvement dwarfs the ≤1.4-point standard deviation.
2. **Fresh multi-seed inference runs** conducted for this revision, both with each task's complete tool set. On **VSP** (full tool set incl. Point/Molmo), three independent full-test-set runs gave 64.45 / 64.73 / 64.55 (**mean 64.58, std 0.14**), with the verification sub-task reaching 0.968 (matching the paper's 99.20-level result). On **Jigsaw** (local-only tool set), three runs gave 88.20 / 88.20 / 88.40 (**mean 88.27, std 0.12**), reproducing the paper's Jigsaw-COCO number (88.60).

Seed-to-seed variance is a small fraction of a point, far below the gains our method produces. Given resource constraints, variance is reported at the **inference level** (fixed checkpoint, stochastic multi-turn decoding); we state this scoping explicitly and do not claim training-seed variance.

## R2-4 — The architecture is not evaluated under tool failure
We add a **controlled fault-injection study** (new Table X): we inject five fault types — plausible-but-wrong output, missing response, malformed response, timeout, and contradictory tools — at **early** turns (round 1), and report the rate at which the model **detects/reacts** (post-fault reflection or tool re-call), **recovers** (final answer still correct), or **propagates** (final answer wrong with no reaction), on fixed 100-item subsets of **two tasks** (VSP, multi-turn planning; Jigsaw, single-shot 3-choice) against no-fault baselines.

**VSP** (baseline acc 0.34):
| Fault (early) | detect/react | recover (acc) | propagate | Δacc |
|---|---|---|---|---|
| plausible-but-wrong | 1.00 | 0.39 | 0.00 | +0.05 |
| missing | 0.99 | 0.36 | 0.01 | +0.02 |
| malformed | 0.99 | 0.29 | 0.01 | −0.05 |
| contradictory | 0.98 | 0.30 | 0.02 | −0.04 |
| timeout | 0.70 | 0.28 | 0.24 | −0.06 |

**Jigsaw** (baseline acc 0.90):
| Fault (early) | detect/react | recover (acc) | propagate | Δacc |
|---|---|---|---|---|
| plausible-but-wrong | 1.00 | 0.77 | 0.00 | −0.13 |
| missing | 1.00 | 0.73 | 0.00 | −0.17 |
| malformed | 1.00 | 0.84 | 0.00 | −0.06 |
| contradictory | 1.00 | 0.82 | 0.00 | −0.08 |
| timeout | 0.92 | 0.81 | 0.03 | −0.09 |

Findings: (1) the model **detects/reacts to the large majority of injected faults**, issuing extra reflection turns and re-calling tools (average turn count rises vs. baseline), with **near-zero propagation** except VSP-timeout; (2) **timeout is the hardest fault** (VSP detect 0.70 / propagate 0.24) because a hard failure returns no content to react to — we flag this as the main robustness gap; (3) a **task-structure contrast**: the multi-turn planning task (VSP) can re-plan, so accuracy degrades gracefully (worst −6), whereas the single-shot task (Jigsaw) detects faults at an even higher rate but, once a corrupted result misleads the one-shot choice, shows larger drops (up to −17) — detection is necessary but recovery headroom depends on task structure. This is direct evidence that the failure/reflection trajectories in cold-start translate into fault-robustness. We note early-vs-late injection as a natural extension (early faults have the most turns for recovery). **[AUTHORS TO FILL (resource-permitting): cold-start ablation with vs. without failure&reflection trajectories, to attribute the recovery behaviour.]**

## R2-5 — Inference cost is never measured
Addressed jointly with R1-6a: we report latency distributions, a generation/execution/orchestration breakdown, per-tool times, and an accuracy-vs-latency curve at a matched budget. Key finding: with **local-only** tools (Jigsaw) generation is 91.0% and tool execution just 0.55%; with an **expert-model tool** invoked many times per sample (VSP with Point/Molmo-7B) tool execution rises to 48.1%. A per-tool micro-benchmark: local operator (A\*) ~0.09 ms/call vs expert-model call (Point) ~255 ms/call (~2,800×). Hence **CPS is not a wall-clock proxy**, and cost-aware invocation of expensive tools is where latency is actually saved. (VSP uses a single Point worker; tool-execution includes queueing and is an upper bound.) **[AUTHORS TO FILL: accuracy-vs-latency curve figure.]**

## R2-6 — Possible image-level leakage in Jigsaw-COCO
We confirm the splits are **image-disjoint at the source-image level**: Jigsaw-COCO puzzles are constructed from COCO source images, and training and test puzzles are built from **different source images**, not from different patch positions of the same image. The 1,000-sample test set is generated only from held-out COCO images that never contribute any patch to training. We clarify Sec C.1 to state this guarantee explicitly. As an additional safeguard we run a **near-duplicate check** between the train and test source images — perceptual-hash pHash (Hamming distance ≤5) plus a CLIP-embedding cosine-similarity screen (≥0.95) — and report the overlap count in the appendix. **[AUTHORS TO FILL: run pHash+CLIP on the full train/test source images and report the true overlap count (expected "0 overlapping pairs"); the local copy has only the test split, so use the full data — do not fabricate.]**

## R2-7 — The LM-judge is under-validated
We strengthen judge validation on V\*, WebMMU, and GUIQA. We draw a random sample of **N=100** items (stratified across the three benchmarks), which **k=2 of the authors** double-blind annotate (blind to system identity and to the LM judge's verdict), and report agreement with the Qwen2.5-VL-72B judge (Cohen's κ, agreement accuracy). We also test for a **length/verbosity bias**: we regress judge score on answer length (tokens) with correctness held fixed, and report the coefficient; controlling for length, judge scores should not increase with verbosity. We report sample size, annotator count, and blinding procedure in full. **[AUTHORS TO FILL, real data only: Cohen's κ and agreement %, and the verbosity-regression coefficient + p-value. Authors' blind annotation is required — do not fabricate. Optionally add a second judge (e.g. GPT-4o) and report inter-judge agreement.]**

## Minor points
- **"Despite never explicitly trained to do so."** We now distinguish *not-supervised-at-the-instance-level* from *not-trained-at-all*, given that Adaptive Reward (Sec 2.3/A.4) and Adaptive Learning (Sec 2.4) shape this behaviour. Wording revised.
- **"Bottleneck shifted from scale to tool quality."** Presented as a **structured-task finding**, noting the smaller general-task gains (~4-7 points on V\*/HRBench/WebMMU vs. ~40 on VSP/Jigsaw) and the "cannot fully offset" caveat.
- **Naming.** We use **GenReasoner** consistently, and update Figures 1 and 10 and the repository so the model is not mistaken for a third-party baseline. We confirm the released code and checkpoints correspond to this manuscript.
- **Reproducibility.** We will release the generated trajectories, exact splits, the VSPO and Jigsaw-COCO construction scripts, and seeds. We correct the cold-start sample accounting. **[AUTHORS TO FILL: decomposition of the 332,649 cold-start samples by task/stage, verified to sum to the total.]**
- **Limitations.** We add a dedicated Limitations section covering latency and cost, dependence on hand-designed trajectories and external expert models, task-specific tooling, and dependence on tool quality (stated as a fragility as well as a strength).
- **Smaller items.** VSPO grid sizes reconciled between A.2 and C.1; task count made consistent (four tasks); table cross-references corrected; terminology unified to "Tool-GRPO"; and we distinguish a *syntactically successful* tool call from a *semantically useful* result.

---

## 附:作者提交前必须补的真实数据(不进提交稿,勿编造)

| # | 位置 | 待补内容 | 成本 |
|---|---|---|---|
| 1 | R1-5 | GUIChat "答对且未调用工具" 子集占比及其正确率 | 已有 rollout,跑一次统计 |
| 2 | R1-6a / R2-5 | 精度–延迟曲线图(vs baseline,匹配预算) | 用已测 latency 数据画图 |
| 3 | R2-4 | 冷启"有/无 failure&reflection 轨迹"消融(资源允许) | 需重训一版对照 |
| 4 | R2-6 | pHash+CLIP 在全量 train/test 源图上的真实重叠计数(预期 0) | 需全量源图(本地只有 test) |
| 5 | R2-7 | Cohen's κ + 一致率;冗长度回归系数+p 值(必须真实盲标) | 作者盲标 N=100,零算力 |
| 6 | Minor | 332,649 冷启样本按任务/阶段分解(须与附录相加吻合) | 核对附录 |

已整合的真实结果(E2 方差 / E3 成本 / E4 reward 自查 / E5 故障注入 / E7 baseline / E8 泄漏声明)均来自 `RESULTS_TABLE.md`,已就位。
