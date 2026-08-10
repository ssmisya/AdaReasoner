# GenReasoner — Point-by-Point Response to Reviewers

We thank the reviewers for identifying the central issues in the submitted
article: the boundary with the ICLR 2026 AdaReasoner paper, the scope of the
generalization evidence, protocol consistency, and the cost and failure surface
of multi-turn tool use. The revision now draws that boundary explicitly and
uses narrower claims wherever the experiment does not isolate the stronger
interpretation.

The revised manuscript is marked in red. AdaReasoner (ICLR 2026) contains the
trajectory-curation pipeline, Tool Cold Start (TC), Tool-GRPO (TG), composite
reward, seven-tool suite, and inherited single-task results. The journal delta
is (i) identifier randomization and description paraphrasing for interface
robustness, (ii) the randomized TC/TG transfer study, and (iii) the broader
V*/HRBench tool-planning evaluation. Revision-time latency, failure, variance,
and judge analyses characterize the system; they are not presented as new
optimization methods.

---

# Reviewer 1

## R1-1: Novelty of the three methodological components

**Comment:** *Trajectory construction, Tool-GRPO, and
randomization/paraphrasing all have substantial precedents.*

The revised paper no longer presents these primitives as individually new.
Cold-start trajectory synthesis and Tool-GRPO are inherited from AdaReasoner
(ICLR 2026), and Tool-GRPO is an instantiation of GRPO for multi-turn external
tool interaction rather than a new RL optimizer. CogCoM/TACO and related
trajectory work are cited as precedents; identifier randomization and
paraphrasing are positioned as robustness-oriented data augmentation.

The journal contribution is narrower and explicit: Adaptive Learning applies
identifier randomization and semantic-preserving documentation paraphrases in
both TC and TG; the randomized training matrix studies where that intervention
helps; and V*/HRBench extend the tool-planning evaluation beyond the original
structured tasks. The Abstract, Introduction, Related Work, Method, captions,
and Conclusion now state this conference/journal boundary.

## R1-2: Function understanding or interface remapping?

**Comment:** *The renamed tools preserve functionality, so the experiment
cannot distinguish abstract function understanding from interface mapping.*

That distinction is now explicit. The randomization study demonstrates
**interface-level robustness**: tool/argument identifiers and descriptions
change while the underlying function is preserved. It does not demonstrate
mastery of an arbitrary new capability.

The separate A* study has a narrower interpretation. A* is absent from TC; in
the inference-only condition it is also absent from TG and is exposed only at
evaluation. Navigation rises from **44.83 to 62.33**, and **94.53%** of calls
execute syntactically. The same intervention lowers verification from **94.20
to 80.00**, because A* is irrelevant there. Thus the result shows that the
model can invoke one useful, cold-start-unseen interface without a matching
demonstration, while also showing that zero-demonstration routing is unstable.
It is not evidence of unrestricted functional abstraction. The revised text
uses these exact boundaries.

The enlarged-pool statistics are consistent with this interpretation: the
model mostly retains familiar tools when redundant alternatives are exposed,
uses complementary A*, and ignores irrelevant RotateImage/GetWeather. These
statistics show task-conditional routing in the tested pool, not a general
sample-complexity law.

## R1-3: Closed-source evaluation protocol

**Comment:** *It is unclear whether proprietary models receive the same tools
and prompts; a tool model versus a no-tool model is not protocol matched.*

The revised evaluation section and table caption now separate the protocols:

- GPT-5, Claude Sonnet 4, Gemini 2.5 Flash, and open-source rows without
  “+Tools” are **no-tool, single-turn** baselines.
- Rows marked “+Tools” receive the same task-specific schemas, multi-turn
  interaction format, and maximum-round budget used in tool-enabled AdaEval.

The matched GPT-5 comparison makes the effect concrete: adding tools improves
GPT-5 from **55.64 to 71.36** on VSP and from **80.10 to 84.50** on Jigsaw.
The trained orchestration model remains stronger on these structured tasks.
The manuscript no longer treats the no-tool proprietary rows as a matched
agent comparison and makes no claim of universal superiority on open-ended
tasks.

## R1-4: DeepEyes / PixelReasoner fairness

**Comment:** *Their low CPS/success could reflect interface incompatibility
rather than inherent inferiority.*

Correct. DeepEyes and PixelReasoner were designed for single-tool or fixed-loop
interaction and were not fine-tuned or prompt-adapted to this multi-tool
interface. Their results show brittleness under an unseen interface, not an
intrinsic upper bound on those methods. The main-table and tool-statistics
captions now state this caveat. We do not claim nonexistent post-adaptation
results. We also distinguish syntactic execution success from semantic utility:
a call can parse and execute while returning a misleading observation.

## R1-5: Asymmetric reward and reward hacking

**Comment:** *The reward lacks a theoretical justification and may encourage
direct answers only when the model is confident.*

The paper now describes the asymmetric term as bounded empirical reward
shaping within standard GRPO; no separate convergence guarantee is claimed.
The inherited reward-weight experiment shows that the tool term is useful:
raising \(\lambda_{tool}:\lambda_{acc}\) from 0:1 to 2:1 changes VSP overall
from **71.45 to 93.27** and VSPO overall from **57.37 to 82.34** under the
reported 100-step setup.

That ablation does **not** establish cost-aware abstention. Full-rollout audits
show that VSP/Jigsaw use tools on nearly all samples, and every GUIChat run uses
at least one tool on **962/962** samples. The manuscript therefore removes the
claim that the reward empirically teaches the model to avoid tools whenever a
direct answer is available. It supports executable tool behavior, not a causal
selective-use policy.

## R1-6: Latency, tool-set scaling, and error propagation

### (a) Latency

The revised appendix reports measured wall-clock decomposition for the 7B
policy on one H20 (TP=1; VSP/VSPO additionally use Point/Molmo):

| Benchmark | generation | tool execution | orchestration | other/I/O | wall time |
|---|---:|---:|---:|---:|---:|
| BLINK-J | 92.48% | 0.39% | 0.01% | 7.11% | 175.9 s |
| Jigsaw | 90.95% | 0.55% | 0.02% | 8.49% | 1,275.8 s |
| GUIChat | 73.85% | 18.65% | 0.01% | 7.50% | 2,744.2 s |
| V* | 31.16% | 48.11% | 0.05% | 20.69% | 1,411.9 s |
| VSPO | 43.83% | 54.50% | 0.01% | 1.66% | 13,427.2 s |
| VSP | 39.11% | 59.42% | 0.01% | 1.47% | 6,541.6 s |

AStar takes **0.092 ms/call**, whereas Point/Molmo takes **255.333
ms/call**, a **2,775×** difference. CPS is therefore not a wall-clock proxy.
These measurements are not a matched-budget accuracy–latency curve, and the
revised paper does not use them to claim universal efficiency.

### (b) Scaling with the exposed tool set

No separate learned router grows with the tool count; schemas are serialized
into the prompt. A larger pool therefore increases schema tokens and makes
autoregressive selection harder. In the tested enlarged pool, calls remain
concentrated on task-relevant tools and the irrelevant GetWeather tool is never
called. One pool is an empirical routing check, not a general complexity
result, and the manuscript says so.

### (c) Intermediate failure

We inject one early fault into fixed 100-item VSP-navigation and 100-item
Jigsaw subsets:

| Fault | VSP (clean 0.34) | Δ | Jigsaw (clean 0.90) | Δ |
|---|---:|---:|---:|---:|
| plausible-but-wrong | 0.39 | +0.05 | 0.77 | -0.13 |
| missing | 0.36 | +0.02 | 0.73 | **-0.17** |
| malformed | 0.29 | -0.05 | 0.84 | -0.06 |
| timeout | 0.28 | **-0.06** | 0.81 | -0.09 |
| contradictory | 0.30 | -0.04 | 0.82 | -0.08 |

The VSP condition omits Point and has a low clean baseline; it must not be
compared to the full-tool VSP headline result. The automatic detect/react
heuristic also fires on clean trajectories, so we report accuracy only. Late
injection and with/without failure-trajectory training ablations are not
claimed as completed.

---

# Reviewer 2

## R2-1: Boundary with the published AdaReasoner paper

The manuscript now cites *AdaReasoner: Dynamic Tool Orchestration for
Iterative Visual Reasoning* (ICLR 2026; OpenReview `nUGPEmQ2ut`) at first
mention and separates inherited content from the journal delta throughout the
Abstract, Introduction, Method, Experiment captions, Discussion, and
Conclusion.

| Component | ICLR 2026 AdaReasoner | Journal extension |
|---|---|---|
| trajectory curation, TC, Tool-GRPO, reward, seven tools | introduced | retained for a self-contained article |
| single-task VSP/Jigsaw/GUIQA study | introduced | inherited context |
| randomized identifiers + description paraphrases | — | **new interface-robustness method** |
| randomized TC/TG transfer matrix | — | **new systematic study** |
| V*/HRBench tool-planning evaluation | — | **new broader evaluation** |
| latency/failure/judge analyses | — | added in revision to characterize the system |

The journal paper no longer re-claims the conference core as new.

## R2-2: Scope of generalization

The two claims are separated.

1. **Tasks:** Jigsaw-only TC with VSP/WebQA introduced in TG is
   **cross-stage transfer**, not zero-shot task-family generalization. V*/HRBench
   are benchmark-level transfer settings, but related visual-search data/tools
   appear during TG; the paper no longer says “no related training.”
2. **Tools:** renamed/rephrased tools preserve functionality and therefore test
   **interface robustness**, not genuinely new capabilities.

The revised headings, table captions, Abstract, Introduction, and Conclusion
use these terms consistently.

## R2-3: Statistical reliability and inconsistent values

The cross-table differences arose from mixed task loaders, judges, prompts,
and tool settings. The revised paper binds each new reliability number to a
single protocol and reports fixed-checkpoint stochastic inference repeats:

| Benchmark | run 1 | run 2 | run 3 | mean ± sample std |
|---|---:|---:|---:|---:|
| VSP | 89.91 | 89.64 | 88.27 | **89.27 ± 0.88** |
| VSPO | 78.98 | 78.32 | 78.62 | **78.64 ± 0.33** |
| Jigsaw-COCO | 88.20 | 88.20 | 88.40 | **88.27 ± 0.12** |
| BLINK-J | 88.00 | 88.67 | 88.00 | **88.22 ± 0.39** |
| V* | 68.59 | 68.06 | 67.54 | **68.06 ± 0.53** |
| GUIChat, 72B judge | 73.70 | 73.49 | 73.60 | **73.60 ± 0.11** |
| WebMMU Functional/Act., 72B judge | 72.15 | 71.14 | 71.95 | **71.75 ± 0.53** |

These are three stochastic inference repeats of one checkpoint at temperature
0.7, not training-seed variance or a significance test. HRBench is deliberately
excluded from this new table: the legacy result contains 108–111 `Z` fallbacks
per run from an unavailable external answer extractor and is not frozen until
uniform offline re-scoring is complete.

## R2-4: Robustness under tool failure

The controlled early-turn results and their limitations are reported under
R1-6(c) and in the revised appendix. They establish a task-dependent accuracy
change, not a 70–100% fault-detection rate. The manuscript does not causally
attribute the observed robustness to reflection/failure trajectories because a
with/without training ablation has not been completed.

## R2-5: Inference cost

The revised appendix reports stage-level wall time and per-tool latency under a
specified hardware/serving configuration. It explicitly states that these data
show cost heterogeneity but do not replace the requested matched-budget curve.
Accordingly, favorable efficiency claims have been removed; the remaining
claim is that orchestration overhead is small relative to generation/tool
execution in the measured system.

## R2-6: Jigsaw source-image leakage

The construction code samples unique COCO source images, partitions those
images into SFT/RL/test lists, and only then constructs puzzles and chooses
missing-patch positions. Derived patches from one source image therefore stay
within one split; lower-right versus other patch positions define a task
condition rather than the split. This guarantee is now stated explicitly in
the appendix, and the revision release will include source-image manifests.

A pHash+CLIP near-duplicate screen requires the complete source-image sets,
which are not present in this checkout. We do not claim a completed
near-duplicate result. Exact source-image disjointness and perceptual
near-duplication are reported as distinct checks.

## R2-7: LM-judge validation

The revised paper reports the actual audit protocol rather than the earlier
unquantified “consistent” statement. We sampled 500 GUIChat/WebMMU records
(seed 260118631), of which 498 have valid references:

| Statistic | Result |
|---|---:|
| agreement | **90.76% (452/498)** |
| Wilson 95% CI | **87.90%–93.00%** |
| Cohen’s κ | **0.781** |
| precision / recall / specificity | **95.59% / 91.29% / 89.44%** |
| FP / FN | **15 / 31** |
| GUIChat agreement | **86.29% (170/197)** |
| WebMMU agreement | **93.69% (282/301)** |

The archive is a reproducible **single-reviewer semantic audit**; it is not
described as a blinded two-human study. Agreement across response-length
quartiles is 87.10/91.20/93.55/91.20%, with no monotonic increase for the
longest quartile. This is descriptive evidence, not a causal proof that
verbosity has no effect.

## Minor points

- “Never explicitly trained” is replaced by stage- and interface-specific
  descriptions.
- “Tools shift the bottleneck from scale” is restricted to the studied
  structured tasks.
- GenReasoner is used consistently for the journal model; AdaReasoner denotes
  the cited conference lineage.
- A dedicated Discussion and Limitations section now covers latency,
  expert-worker dependence, hand-designed trajectories, interface-level scope,
  and tool-quality fragility.
- The appendix distinguishes syntactic execution from semantic utility and
  uses one four-task-family description.
- `332,649` is now identified as the configured `max_samples` cap, not a
  verified count of unique trajectories. The release manifest will provide the
  actual post-filter task decomposition.
- The release commitment now covers trajectories, source-image/task manifests,
  construction scripts, prompts/rewards, tool schemas, checkpoint identifiers,
  and inference metadata.

---

# Closing Response

The revision makes one argument throughout: dynamic tool orchestration can
produce large gains when an external operator supplies a useful missing
capability, but those gains must be interpreted together with the training
stage, interface relationship, tool reliability, and test-time cost. The paper
now separates inherited conference contributions from the journal delta,
scopes every generalization claim to the actual protocol, and reports the
failure and cost surface rather than presenting tool augmentation as removing
fallibility.
