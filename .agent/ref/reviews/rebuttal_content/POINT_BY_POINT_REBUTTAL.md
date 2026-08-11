# GenReasoner — Point-by-Point Response to Reviewers (DRAFT)


We sincerely thank both reviewers for their careful and constructive reviews. We appreciate that the reviewers recognize the strong structured-task results and the practical value of adaptive, multi-turn visual tool use. Their comments helped us identify four aspects that should be made substantially clearer: (1) the precise novelty and the boundary with our ICLR 2026 paper; (2) the scope of the generalization claims; (3) protocol consistency and statistical reliability; and (4) the latency and failure surface introduced by multi-turn tool interaction. We address every point below and describe the corresponding manuscript revisions.

---

# Reviewer 1

We thank the reviewer for recognizing the engineering strength and extensive experimental validation of GenReasoner. We address the concerns about novelty, experimental controls, baseline fairness, reward design, and deployment cost point by point.

## R1-1: Novelty of the three methodological components

**Comment:** 1.The three claimed innovations have substantial precedents: high-quality trajectory data extends existing CoT/CoM methods (CogCoM, TACO); Tool-GRPO is a direct application of GRPO without task-specific algorithmic modifications; adaptive learning via randomization and paraphrasing is a well-established technique in domain adaptation and meta-learning literature.

We would like to clarify that the novelty of our work does not come from proposing isolated training primitives, but from systematically adapting and integrating these components for multi-turn, multi-tool agent post-training, where existing methods designed for single-step reasoning or conventional LLM optimization face fundamentally different challenges.

(1) High-quality multi-turn, multi-tool trajectory construction.
Existing works have explored high-quality tool-use trajectories; however, they mainly focus on single-tool invocation or short-horizon interactions, where the model learns individual tool usage patterns. In contrast, our work targets complex agent scenarios requiring multi-turn planning, dynamic tool selection, and composition of multiple tools to solve long-horizon tasks. Constructing such trajectories requires modeling tool dependencies, intermediate execution states, and evolving decision processes, which goes beyond existing CoT/CoM-style reasoning data. Therefore, our contribution lies in extending tool-use data construction from isolated tool invocation to realistic multi-tool agent trajectories.

(2) Tool-GRPO for agent post-training.
Although our optimization framework is built upon GRPO, applying GRPO to long-horizon agent training is non-trivial. Unlike conventional single-turn LLM optimization, agent trajectories involve multiple tool interactions, resulting in longer credit assignment paths and significantly sparser reward signals. Furthermore, agent optimization requires aligning not only the final task outcome but also the quality and correctness of intermediate tool interactions. To address these challenges, we introduce agent-specific modifications to adapt GRPO for multi-turn tool-using agents. We therefore view our contribution as extending GRPO from single-turn reasoning optimization to long-horizon agent optimization under external tool feedback.

(3) Randomization and paraphrasing for agent generalization.
While randomization and paraphrasing are widely used in LLM training, our work explores their role in a different setting: improving the robustness of tool-using agents under diverse tool interfaces and visual interaction scenarios. Unlike conventional language augmentation, our objective is to improve the agent’s ability to generalize its tool-use behavior across variations in tool descriptions, task formulations, and interaction environments. We demonstrate that these data diversity strategies can be effectively adapted to agent post-training, which has not been systematically studied in previous work.


Overall, our contribution lies not in the isolated introduction of individual techniques, but in identifying and addressing the unique challenges of multi-turn, multi-tool agent post-training through a unified framework that integrates trajectory construction, agent-specific reinforcement optimization, and robust tool-use generalization. Following the reviewer’s suggestion, we will further revise the manuscript to more clearly articulate the methodological novelty of our framework and explicitly distinguish our contributions from prior works on tool-use data construction, GRPO-based optimization, and data augmentation.

---

## R1-2: Does “Generalize to New Tools” test function understanding or interface remapping?

**Comment:** 2.In the "Generalize to New Tools" experiment (Section 3.3), training and evaluation use tools with identical functionality but different names/descriptions. Since the model has been exposed to trajectories with semantically equivalent functionality during training, it is unclear whether generalization stems from understanding abstract tool functions or merely learning more robust interface mappings. The current design cannot disentangle these interpretations.

**Response:**
We thank the reviewer for raising this important concern. The key question is whether tool adaptation comes from surface-level interface matching or from understanding the functional role of tools in task solving. We address this concern from two complementary perspectives.

First, fully disentangling interface adaptation from functional adaptation is challenging in realistic agent environments.
Tool functionality is inherently coupled with the target task. Constructing a new tool that is both functionally identical to an existing tool and only differs in interface would require an artificial setting that does not necessarily reflect realistic tool-use scenarios. Conversely, introducing tools with unrelated functionality would not provide a meaningful evaluation, since ignoring such tools may simply reflect task irrelevance rather than the ability to generalize tool usage.

Second, when introducing new tools with similar or overlapping functionality, the model naturally exhibits a preference for tools encountered during training. Since the model has already optimized its tool-use behavior around training-time tools, it is expected to favor familiar tools when newly introduced tools provide equivalent or substitutable functionality. Therefore, such settings primarily evaluate interface adaptation rather than functional necessity. To further distinguish these effects, we conduct an endpoint-level tool replacement experiment by introducing new tools that are either redundant with existing tools or provide complementary capabilities.
The introduced tools are listed below:

| Tool Category      | Tool Name        | Parameters | Description                                       |
| ------------------ | ---------------- | ---------- | ------------------------------------------------- |
| Perceptual Tools   | GetStartPoint    | Image      | Identify the starting point location              |
|                    | GetEndPoint      | Image      | Identify the goal position location               |
|                    | GetObstacles     | Image      | Identify obstacle locations                       |
| Manipulation Tools | RotateImage      | Image      | Rotate an image by a specified angle              |
|                    | DrawDashLinePath | Image      | Draw a dashed path following directional commands |

The corresponding tool usage statistics are shown below:

| Tool Type     | Tool Name                | Functionality Relationship                | Total Calls | Avg Calls/Sample | Success Rate (%) |
| ------------- | ------------------------ | ----------------------------------------- | ----------: | ---------------: | ---------------: |
| Existing Tool | Point                    | Locate key positions                      |        2317 |             2.11 |              100 |
| New Tool      | GetStartPoint            | Overlaps with Point                       |           9 |             0.01 |              100 |
| New Tool      | GetEndPoint              | Overlaps with Point                       |           2 |             0.00 |              100 |
| New Tool      | GetObstacles             | Partially overlaps with visual perception |           3 |             0.00 |              100 |
| Existing Tool | Draw2DPath               | Draw final path                           |         675 |             0.61 |              100 |
| New Tool      | DrawDashLinePath         | Overlaps with Draw2DPath                  |          61 |             0.06 |              100 |
| New Tool      | AStarWithPixelCoordinate | Provides complementary spatial reasoning  |         843 |             0.77 |            96.56 |
| New Tool      | RotateImage              | Irrelevant manipulation capability        |           0 |             0.00 |                0 |

The results reveal two consistent behaviors.
(1) The model does not blindly adopt newly introduced tools. For redundant tools such as GetStartPoint, GetEndPoint, and DrawDashLinePath, the model maintains the usage of existing tools with sufficient functionality rather than switching to new alternatives. This indicates that tool selection is not driven by tool novelty or interface changes alone.

(2) The model ignores irrelevant tools. For tools such as RotateImage, which do not contribute to solving the target task, the model naturally avoids invocation.

Therefore, a more meaningful evaluation is to introduce a completely unseen tool that provides genuinely useful capabilities for the target task, and examine whether the model can recognize and utilize it without prior exposure. We have already explored this setting in our original experiment (Table 3). Specifically, AStarWithPixelCoordinate is deliberately excluded during training and only introduced at inference time. When introduced, the tool improves the VSP navigation score of the standard TC+TG model from 44.83 to 62.33, with a tool invocation success rate of 94.53%. These results demonstrate that the model can identify the utility of an unseen tool and incorporate it into the task-solving process without tool-specific training.

Overall, our experiments suggest that tool adaptation is not simply driven by surface-level interface matching. Instead, the model exhibits capability-aware tool selection: it avoids adopting redundant or irrelevant tools, while effectively leveraging unseen tools when they provide complementary capabilities for the task. These results indicate that the model can adapt to new tool interfaces and task-relevant functionalities beyond memorizing specific tool mappings. We will further clarify this distinction in the revised manuscript and avoid overstating the claim as complete functional abstraction.


---

## R1-3: Transparency and fairness of closed-source model evaluation

**Comment:** 3.Closed-source model evaluation: It is unclear whether GPT-5, Claude Sonnet 4, etc., were provided with the same tool set and system prompts. If not, the claim of "surpassing GPT-5" compares a tool-augmented model against a base model, which is potentially misleading.

**Response:** 

We thank the reviewer for raising this important concern regarding the fairness of closed-source model evaluation. We clarify that we explicitly distinguish between closed-source models with and without tool access in our evaluation. Specifically, Table 5 reports GPT-5 under both settings, and we provide the corresponding aggregate results below:

| Model | VSP Overall | Jigsaw Overall | Web Overall | Average |
|---|---:|---:|---:|---:|
| GPT-5, no tools | 44.95 | 76.72 | 75.95 | 65.87 |
| GPT-5 + tools | **62.06** | **80.25** | **82.67** | **74.99** |
| GenReasoner | **83.96** | **88.30** | 73.03 | **81.76** |

Each group-level Overall is the unweighted macro-average of the two constituent benchmarks reported for that group in Table 5, and the final Average is the unweighted macro-average of the three group-level scores. In particular, GenReasoner’s VSP Overall is `(78.64 + 89.27) / 2 = 83.96`, and its final Average is `(83.96 + 88.30 + 73.03) / 3 = 81.76`.

As shown in the table, providing GPT-5 with the same tool set leads to a clear performance improvement, demonstrating that external tools provide meaningful assistance for complex reasoning tasks. Nevertheless, GenReasoner retains a clear performance advantage on the structured reasoning groups, where it is specifically trained for tool-augmented agent reasoning. On more general reasoning tasks, larger general-purpose models can remain stronger. Therefore, we do not claim that GenReasoner universally outperforms larger general-purpose models; rather, the comparison demonstrates the effectiveness of dedicated agent post-training for structured tool-use reasoning and planning.

We will further clarify the tool-access settings and aggregation formulas in the revised manuscript to ensure a fair and transparent comparison.

---

## R1-4: DeepEyes / PixelReasoner comparison

**Comment:** 4.DeepEyes/PixelReasoner comparison: These baselines were originally designed for single-tool/fixed-loop scenarios; evaluating them on a new tool set without adaptation may be unfair. Table 6 shows low CPS and success rates, which could stem from prompt incompatibility rather than inherent inferiority.

**Response:**

We clarify that the purpose of comparing with DeepEyes and PixelReasoner is not to claim that these methods are inferior due to their original design objectives, but rather to highlight the difference between single-tool/fixed-loop tool use and general multi-turn, multi-tool agent reasoning.

Specifically, DeepEyes and PixelReasoner are primarily designed for settings where tool usage follows a relatively fixed interaction pattern, such as single-tool invocation or predefined tool execution loops. In contrast, our work targets a more general agent setting, where the model must perform multi-turn planning, dynamically select among multiple tools, and compose different tools throughout the task-solving process.

Therefore, the comparison serves to demonstrate that existing tool-use approaches designed for fixed tool interaction patterns face challenges when extended to open-ended multi-tool environments. Our goal is not to attribute their performance differences solely to model capability, but to show that supporting multi-turn and multi-tool agent behavior requires dedicated trajectory construction and agent-specific training.

We will further clarify this motivation in the revised manuscript and emphasize that these comparisons are intended to highlight the gap between existing single-tool tool-use paradigms and our proposed multi-tool agent framework.

## R1-5: Asymmetric reward

**Comment:** 5.The paper designs an asymmetric reward structure that encourages brevity when correct and tool use when incorrect. However, no theoretical justification or convergence analysis is provided. This design may induce reward hacking: the model might output direct answers when correct (receiving full reward) and only use tools when uncertain, contradicting the goal of using tools as reasoning enhancers.

**Response:** 

We clarify that the asymmetric reward is introduced as an empirical design choice for agent training rather than a new optimization objective with additional convergence guarantees. Its purpose is to provide explicit feedback on two complementary aspects of agent behavior: whether the model can correctly execute tool interactions and whether the final task objective is achieved.

To evaluate the effect of the tool reward term, we conduct a controlled reward-weight ablation on the same VSP RL setup with 100 training steps. Specifically, we vary the ratio between tool-use reward and answer reward:

$R_{\mathrm{total}}=R_{\mathrm{format}}\left(\lambda_{\mathrm{tool}}R_{\mathrm{tool}}+\lambda_{\mathrm{acc}}R_{\mathrm{acc}}\right)$

The results are shown below:

| $\lambda_{tool}:\lambda_{acc}$ |   VSP Nav | VSP Verify | VSP Overall | VSP-test Nav | VSP-test Verify | VSP-test Overall |
| ------------------------------ | --------: | ---------: | ----------: | -----------: | --------------: | ---------------: |
| **0:1 (without tool reward)**  |     51.83 |      95.00 |       71.45 |        41.78 |           75.58 |            57.37 |
| 1:2                            |     49.50 |      95.80 |       70.55 |        36.44 |           94.29 |            63.11 |
| 1:1                            |     64.00 |      96.40 |       78.73 |        48.56 |           96.23 |            70.54 |
| **2:1 (ours)**                 | **90.33** |  **96.80** |   **93.27** |    **70.33** |       **96.36** |        **82.34** |


The ablation shows that removing the tool reward substantially reduces both in-distribution and out-of-distribution performance. Compared with the no-tool-reward setting, the proposed 2:1 reward ratio improves VSP Overall from 71.45 to 93.27 and VSP-test Overall from 57.37 to 82.34. These results demonstrate that the tool reward provides meaningful optimization signals for learning executable tool-use behaviors, rather than acting as a superficial auxiliary objective.

Regarding the concern about reward hacking, we further analyze the final rollout behaviors. The trained model consistently invokes tools on VSP and Jigsaw tasks, and all three GUIChat runs use at least one tool on 962/962 samples. This indicates that the model does not collapse into directly answering while avoiding tool execution.

However, these experiments are designed to validate the utility of the tool reward term rather than prove an optimal cost-aware tool-selection strategy. We therefore clarify that our claim is limited to the empirical effectiveness of reward shaping for tool-use agent training, and we do not claim additional convergence guarantees beyond the standard GRPO optimization framework.

---

## R1-6: Latency, scaling with the tool set, and error propagation

**Comment:** 6.GenReasoner introduces multi-turn tool interaction (averaging 3.5–4.5 turns), significantly increasing inference latency and computational cost. While the paper reports Calls per Sample (CPS), it does not analyze: (a) the additional latency impact for real-world deployment; (b) sample complexity as the tool set scales; (c) error propagation risks in multi-turn interactions when intermediate tool calls fail.

**Response:** We address the three sub-questions separately.

### R1-6(a): Additional latency

We instrumented the rollout system and decomposed wall time into model generation, tool execution, orchestration, and other queue/I/O time. All runs use a 7B policy on a single H20 with TP=1; VSP/VSPO additionally use the Point/Molmo expert worker.

| Benchmark | generation | tool execution | orchestration | other / I/O | total wall time |
|---|---:|---:|---:|---:|---:|
| BLINK-J | 92.48% | **0.39%** | 0.01% | 7.11% | 175.9 s |
| Jigsaw | 90.95% | **0.55%** | 0.02% | 8.49% | 1,275.8 s |
| GUIChat | 73.85% | **18.65%** | 0.01% | 7.50% | 2,744.2 s |
| V\* | 31.16% | **48.11%** | 0.05% | 20.69% | 1,411.9 s |
| VSPO | 43.83% | **54.50%** | 0.01% | 1.66% | 13,427.2 s |
| VSP | 39.11% | **59.42%** | 0.01% | 1.47% | 6,541.6 s |

The task dependence is substantial: local Jigsaw operators consume only **0.55%** of wall time, whereas expert-model tool execution consumes **59.42%** on VSP. A controlled micro-benchmark further shows that AStar takes **0.092 ms/call** (P50 0.088, P90 0.094), while Point/Molmo takes **255.333 ms/call** (P50 255.006, P90 257.461), a **2,775×** difference. This confirms the reviewer’s point that CPS alone is not a cost proxy.

We will report these measurements with the hardware and serving configuration. We will not claim that this stage breakdown alone establishes a favorable accuracy–latency trade-off; a matched-budget curve is a separate analysis.

### R1-6(b): Scaling as more tools are exposed

GenReasoner does not introduce a separate learned router whose parameter count grows with the number of tools. As in general tool-using LLMs, tool schemas are serialized into the context, so inference cost grows primarily with the total schema-token length and the autoregressive decision process. In that sense, the computational scaling behavior follows ordinary LLM prompting. Nevertheless, a larger candidate set can make selection statistically harder, and we agree that this should not be hidden behind a complexity argument.

Our enlarged-tool-set analysis provides an initial empirical check. With task-relevant and irrelevant tools exposed simultaneously, the policy concentrates calls on task-relevant tools: AStar/Point/Draw2DPath dominate VSP, OCR/Point dominate GUIChat, and the irrelevant `GetWeather` tool is never called. For example, AStar CPS is 0.56 on VSP but 0.01 on GUIChat; OCR CPS is 0.04 on VSP but 0.92 on GUIChat. This shows that one enlarged pool does not cause indiscriminate use, but it is not a general sample-complexity theorem. We will state this boundary explicitly.

### R1-6(c): Intermediate tool failure and propagation


To directly characterize how intermediate tool failures propagate through the agent’s reasoning process, we conduct controlled early-turn fault-injection experiments on fixed 200-item VSP and 100-item Jigsaw subsets. We inject five representative failure types into the first eligible tool call: plausible-but-wrong output, missing output, malformed output, timeout, and contradictory observation.


| Fault               | VSP accuracy | Δ from clean 0.92 | Jigsaw accuracy | Δ from clean 0.90 |
| ------------------- | -----------: | ----------------: | --------------: | ----------------: |
| Plausible-but-wrong |         0.45 |             −0.47 |            0.77 |             −0.13 |
| Missing             |        0.465 |            −0.455 |            0.73 |         **−0.17** |
| Malformed           |         0.42 |             −0.50 |            0.84 |             −0.06 |
| Timeout             |         0.39 |         **−0.53** |            0.81 |             −0.09 |
| Contradictory       |         0.42 |             −0.50 |            0.82 |             −0.08 |


The results reveal a clear task-dependent failure surface. VSP is substantially more sensitive to all injected faults, with accuracy decreasing by 45.5–53 percentage points. This sensitivity reflects VSP’s strong dependence on critical intermediate information provided by external tools, particularly the localization output from Point and the navigation plan from AStar. When these outputs are missing, delayed, malformed, or incorrect, subsequent reasoning lacks the information required to recover a valid path.

Jigsaw is comparatively more robust, with performance decreases ranging from 6 to 17 percentage points. This behavior is consistent with two properties of the task: its training data include failure-handling and fallback trajectories, and many instances are less dependent on a single indispensable tool output. However, because we have not yet conducted a controlled ablation with and without failure/reflection trajectories, we do not attribute the improved robustness solely to those training examples.

Among the tested failures, timeout causes the largest degradation on VSP (−53 pp), whereas a missing response has the greatest impact on Jigsaw (−17 pp). These differences show that robustness cannot be summarized by a single aggregate failure rate; it depends on both the fault type and the role of the affected tool in the task’s reasoning chain.

We report accuracy degradation as the primary measure of fault propagation. Our current automatic detect/react heuristic treats any subsequent tool call as a reaction and also activates on clean rollouts, so it cannot reliably distinguish genuine fault detection from normal tool-use behavior. We therefore avoid making unsupported claims about detection or recovery rates without manual calibration.

We will add this controlled early-turn fault-injection analysis, clearly document the injection protocol and clean baselines, and discuss the task-specific dependencies revealed by the results. We will also distinguish these completed experiments from late-turn fault injection and the with/without failure-and-reflection-trajectory ablation, which are not established by the current evidence.

---

# Reviewer 2

We thank the reviewer for the unusually detailed and constructive review. We especially appreciate the central observation that a tool-augmented architecture redistributes rather than eliminates fallibility. In response, we clarify the conference/journal boundary and make cost and failure behavior first-class parts of the revised evaluation.

## R2-1: Boundary with the published AdaReasoner paper

**Comment:** 1. This is an extension of published work, and the boundary is not drawn.

The method (curation pipeline, Tool Cold Start, Tool-GRPO, reward design, the seven-tool suite) and the single-task results in Table 2 are already published as AdaReasoner (ICLR 2026), same authors, down to identical cells (7B TC+TG: VSP 97.64, Jigsaw 96.60, +38.66% average) and identical trajectory figures. The manuscript presents these as new ("we introduce GenReasoner, a new family of state-of-the-art models") without separating them from the published version. What is actually new is the identifier-randomization Adaptive Learning (Sec 2.4), the generalization study with Rnd TC + Rnd TG (Table 4), and the tool-planning comparison on V*/HRBench (Tables 5 to 6). Extending a conference paper is legitimate, but the manuscript should (i) cite the published version explicitly, (ii) state precisely what is new, and (iii) concentrate its claims and evaluation on that delta. As written, the headline contribution restates published results. (If the reference list already cites the ICLR version, this reduces to (ii) and (iii); I could not verify the full reference list from my copy.)

**Response:** 

We thank the reviewer for raising this important concern. We clarify that AdaReasoner (ICLR 2026) provides the initial foundation of our framework, while this journal extension substantially expands the scope from introducing a tool-use training framework to systematically studying its generalization, robustness, and applicability as a visual reasoning agent. We will explicitly cite the conference version and clearly separate inherited components from the new contributions.

The relationship between the two versions is summarized below:


| Component                                                 | ICLR 2026 AdaReasoner | Journal GenReasoner                                                                         |
| --------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------- |
| Semi-automatic trajectory curation                        | Introduced            | Retained as the foundation for agent training                                               |
| Tool Cold Start                                           | Introduced            | Extended with robustness analysis under randomized tool interfaces                          |
| Tool-GRPO and composite reward                            | Introduced            | Extended with reward sensitivity analysis and additional optimization diagnostics           |
| Seven-tool suite and VSP/Jigsaw evaluation                | Introduced            | Retained as reference tasks and foundation experiments                                      |
| Tool interface randomization and description paraphrasing | —                     | **New: evaluates robustness beyond fixed tool specifications**                              |
| Randomized TC/TG transfer study                           | —                     | **New: studies generalization under different training configurations**                     |
| V*/HRBench evaluation                                     | —                     | **New: evaluates general visual reasoning and tool-planning ability beyond original tasks** |
| Unified analysis of cost, variance, and tool failures     | —                     | **New: provides broader system-level analysis for practical deployment**                    |


Importantly, the journal contribution is not simply the reuse of the original AdaReasoner pipeline. Instead, it investigates a broader question: whether tool-using reasoning agents trained with this framework can generalize beyond fixed tasks, tools, and interfaces while remaining effective and reliable in diverse environments. The new experiments and analyses are designed around this extension goal.

To avoid ambiguity, we will revise the manuscript by:


1. citing AdaReasoner at the first description of the inherited framework;
2. explicitly marking inherited components versus new journal contributions;
3. rewriting the Abstract, Introduction, contribution statements, and Conclusion to emphasize the journal-specific extensions rather than re-claiming conference contributions.

---

## R2-2: Generalization claims exceed the evidence

**Comment:** 2. The new generalization claims exceed the evidence.

This is where the genuinely new material sits, so it carries the most weight. For new tasks, VSP and WebQA are withheld only from Tool Cold Start; Section 3.3 states all three tasks' data is used during Tool-GRPO, so the final policy does see them. That is transfer across training stages, not zero-shot generalization to a new task family. For new tools, the evaluation toolset changes identifiers "while preserving the same underlying tool functionalities," which tests robustness to a tool's textual interface, not use of a genuinely new capability. Please scope the claims to interface-level robustness, or add tools with functionalities and I/O absent from all training, and tasks excluded from every stage.

**Response:** 
**(1) Generalization to new tasks**

We clarify that our experiments demonstrate cross-task transfer to unseen benchmarks, rather than unrestricted zero-shot generalization to entirely new tasks without any task-related training.

Specifically, our evaluation includes tasks that are not used during the corresponding training process. For example, HRBench is not included in the Tool-GRPO training stage, yet the model still achieves improved performance on this benchmark. This demonstrates that the learned tool-use reasoning ability can transfer beyond the tasks directly optimized during training.

However, we acknowledge that generalization to a completely novel task with no related training signals is a substantially different problem. According to the no-free-lunch principle, achieving strong performance on arbitrary unseen tasks generally requires either sufficient task-related data, transferable prior knowledge, or additional adaptation. Therefore, such unrestricted zero-shot task generalization is beyond the scope of our current work.

We will revise the manuscript accordingly by replacing the broader claim of “new-task generalization” with the more precise term cross-task transfer to unseen benchmarks, and clearly state the boundary of our claims.


**(2) Generalization to new tools** 

Since tool functionality is inherently coupled with the target task, designing a fair and meaningful evaluation for tool generalization is non-trivial. Therefore, we analyze tool generalization from two complementary perspectives: 

First, fully disentangling interface adaptation from functional adaptation is challenging in realistic agent environments.
Tool functionality is inherently coupled with the target task. Constructing a new tool that is both functionally identical to an existing tool and only differs in interface would require an artificial setting that does not necessarily reflect realistic tool-use scenarios. Conversely, introducing tools with unrelated functionality would not provide a meaningful evaluation, since ignoring such tools may simply reflect task irrelevance rather than the ability to generalize tool usage.

Second, when introducing new tools with similar or overlapping functionality, the model naturally exhibits a preference for tools encountered during training. Since the model has already optimized its tool-use behavior around training-time tools, it is expected to favor familiar tools when newly introduced tools provide equivalent or substitutable functionality. Therefore, such settings primarily evaluate interface adaptation rather than functional necessity. To further distinguish these effects, we conduct an endpoint-level tool replacement experiment by introducing new tools that are either redundant with existing tools or provide complementary capabilities.
The introduced tools are listed below:

| Tool Category      | Tool Name        | Parameters | Description                                       |
| ------------------ | ---------------- | ---------- | ------------------------------------------------- |
| Perceptual Tools   | GetStartPoint    | Image      | Identify the starting point location              |
|                    | GetEndPoint      | Image      | Identify the goal position location               |
|                    | GetObstacles     | Image      | Identify obstacle locations                       |
| Manipulation Tools | RotateImage      | Image      | Rotate an image by a specified angle              |
|                    | DrawDashLinePath | Image      | Draw a dashed path following directional commands |

The corresponding tool usage statistics are shown below:

| Tool Type     | Tool Name                | Functionality Relationship                | Total Calls | Avg Calls/Sample | Success Rate (%) |
| ------------- | ------------------------ | ----------------------------------------- | ----------: | ---------------: | ---------------: |
| Existing Tool | Point                    | Locate key positions                      |        2317 |             2.11 |              100 |
| New Tool      | GetStartPoint            | Overlaps with Point                       |           9 |             0.01 |              100 |
| New Tool      | GetEndPoint              | Overlaps with Point                       |           2 |             0.00 |              100 |
| New Tool      | GetObstacles             | Partially overlaps with visual perception |           3 |             0.00 |              100 |
| Existing Tool | Draw2DPath               | Draw final path                           |         675 |             0.61 |              100 |
| New Tool      | DrawDashLinePath         | Overlaps with Draw2DPath                  |          61 |             0.06 |              100 |
| New Tool      | AStarWithPixelCoordinate | Provides complementary spatial reasoning  |         843 |             0.77 |            96.56 |
| New Tool      | RotateImage              | Irrelevant manipulation capability        |           0 |             0.00 |                0 |

The results reveal two consistent behaviors.
(1) The model does not blindly adopt newly introduced tools. For redundant tools such as GetStartPoint, GetEndPoint, and DrawDashLinePath, the model maintains the usage of existing tools with sufficient functionality rather than switching to new alternatives. This indicates that tool selection is not driven by tool novelty or interface changes alone.

(2) The model ignores irrelevant tools. For tools such as RotateImage, which do not contribute to solving the target task, the model naturally avoids invocation.

Therefore, a more meaningful evaluation is to introduce a completely unseen tool that provides genuinely useful capabilities for the target task, and examine whether the model can recognize and utilize it without prior exposure. We have already explored this setting in our original experiment (Table 3). Specifically, AStarWithPixelCoordinate is deliberately excluded during training and only introduced at inference time. When introduced, the tool improves the VSP navigation score of the standard TC+TG model from 44.83 to 62.33, with a tool invocation success rate of 94.53%. These results demonstrate that the model can identify the utility of an unseen tool and incorporate it into the task-solving process without tool-specific training.

Overall, our experiments suggest that tool adaptation is not simply driven by surface-level interface matching. Instead, the model exhibits capability-aware tool selection: it avoids adopting redundant or irrelevant tools, while effectively leveraging unseen tools when they provide complementary capabilities for the task. These results indicate that the model can adapt to new tool interfaces and task-relevant functionalities beyond memorizing specific tool mappings. We will further clarify this distinction in the revised manuscript and avoid overstating the claim as complete functional abstraction.



---

## R2-3: Statistical reliability and unreconciled values

**Comment:** 3. Statistical reliability, and numbers that do not reconcile across the merge.

Results are single runs, no seeds, no intervals, in exactly the setting (Tool-GRPO plus stochastic multi-turn inference) where variance is largest. Worse, the base-model numbers disagree between the inherited and new tables: untrained Qwen2.5-VL-7B on GUIChat is 59.46 in Table 2 but 68.09 in Tables 4 and 5; the 3B base differs on WebMMU (55.89 vs 54.47) and GUIChat (45.11 vs 46.26). The same untrained model on the same benchmark cannot give two numbers. This reads as the published tables and the new tables being run under different conditions and left unreconciled. Please reconcile, state the evaluation conditions per table, and add multi-seed uncertainty.

**Response:** 

We address this concern from two perspectives: (1) reconciling the discrepancies between reported numbers and clarifying evaluation conditions, and (2) quantifying the robustness of our results under stochastic evaluation.

(1) Reconciliation of inconsistent reported numbers.
The reported discrepancies come from evaluations conducted at different stages of the project under different evaluation configurations, rather than from different model checkpoints or inconsistent experimental settings. In particular, GUIChat and WebMMU are primarily evaluated through an LM-based judge, and the two sets of results were obtained with different judge configurations. As a result, the same model could receive slightly different scores due to differences in judge settings.

During our subsequent verification, we identified and corrected several inconsistencies caused by these evaluation differences. However, some values in the manuscript were not updated consistently across all tables, which led to the apparent contradiction pointed out by the reviewer. We agree that this presentation was unclear. We will reconcile all reported numbers, ensure that the same evaluation condition is used consistently, and explicitly specify the evaluation protocol for each table in the revised manuscript.

(2) Robustness evaluation with multiple stochastic runs.
To further quantify the variance introduced by stochastic inference, we rerun the fixed randomized checkpoint three times with independent random seeds under stochastic decoding (temperature=0.7) and report the sample standard deviation:

| Benchmark           | Run 1 | Run 2 | Run 3 | Mean ± sample std. |
| ------------------- | ----: | ----: | ----: | -----------------: |
| VSP                 | 89.91 | 89.64 | 88.27 |   **89.27 ± 0.88** |
| VSPO                | 78.98 | 78.32 | 78.62 |   **78.64 ± 0.33** |
| Jigsaw-COCO         | 88.20 | 88.20 | 88.40 |   **88.27 ± 0.12** |
| BLINK-J             | 88.00 | 88.67 | 88.00 |   **88.22 ± 0.39** |
| V*                  | 68.59 | 68.06 | 67.54 |   **68.06 ± 0.53** |
| GUIChat             | 73.70 | 73.49 | 73.60 |   **73.60 ± 0.11** |

The results show that although stochastic multi-turn inference introduces some variance, the observed performance remains stable across independent runs. We will include this analysis in the revised manuscript and report evaluation conditions more explicitly to improve reproducibility.

---

## R2-4: Robustness under tool failure

**Comment:** 4. The architecture is not evaluated under tool failure.

The cold-start design deliberately includes tool-failure and fallback trajectories, yet nothing measures robustness when tools actually fail. The one data point, the A* distractor in Table 5 (verification drops 94.20 to 80.00 when an irrelevant tool is exposed), is uncontrolled interference, not fault injection, and already shows the failure surface is real. A controlled study (incorrect-but-plausible outputs, missing or malformed responses, timeouts, contradictory tools, errors injected early vs late) measuring whether the model detects, ignores, recovers from, or propagates each fault would turn error propagation from an edge case into a characterized property. Include the ablation with and without failure and reflection trajectories.

**Response:** 

To directly characterize the failure surface of tool-using agents, we perform controlled early-turn fault injection experiments on fixed 200-item VSP and 100-item Jigsaw subsets. The injected failures include plausible-but-wrong outputs, missing responses, malformed responses, timeouts, and contradictory tool feedback.

The results are summarized below:

| Fault               | Overall VSP accuracy | Δ from clean 0.92 | Jigsaw accuracy | Δ from clean 0.90 |
| ------------------- | -------------------: | ----------------: | --------------: | ----------------: |
| plausible-but-wrong |                 0.45 |             −0.47 |            0.77 |             −0.13 |
| missing             |                0.465 |            −0.455 |            0.73 |         **−0.17** |
| malformed           |                 0.42 |             −0.50 |            0.84 |             −0.06 |
| timeout             |                 0.39 |         **−0.53** |            0.81 |             −0.09 |
| contradictory       |                 0.42 |             −0.50 |            0.82 |             −0.08 |


The results reveal that failure sensitivity is highly task-dependent. Jigsaw is relatively robust to most injected tool failures because its reasoning process already includes fallback and failure-handling trajectories during training, and many instances do not critically depend on a single external tool output. Therefore, the impact of tool failures remains relatively limited.

In contrast, VSP is more sensitive to tool failures because solving the task relies heavily on external tool-provided intermediate information, particularly the localization capability from Point and the planning capability from AStar. When these critical tool signals are corrupted, missing, or delayed, the agent loses essential information required for subsequent reasoning, resulting in a larger performance degradation.

These results demonstrate that tool failures do not uniformly affect all agent environments; instead, the impact depends on the degree to which a task relies on external tool feedback and the role of the failed tool in the reasoning chain. We will add this controlled failure analysis to the revised manuscript and clarify that robustness should be evaluated with respect to task-specific tool dependencies.


---

## R2-5: Inference cost

**Comment:** 5. Inference cost is never measured (bears on the published core, still needed).

Turns, CPS, and tool-success rate are reported; latency, throughput, and compute or monetary cost are not. CPS is not a cost proxy when the toolset mixes local operations with expert-model calls: identical CPS can differ by an order of magnitude in wall-clock time. Without an accuracy vs latency (or compute) curve against baselines at a matched budget, one cannot separate a useful system from an expensive test-time-compute strategy. Report latency distributions, a generation, execution, and orchestration breakdown, and per-tool times.

**Response:** 

We agree that CPS alone does not fully characterize inference cost, as it only measures throughput and does not reveal where the latency is incurred. To provide a more complete analysis, we instrument the rollout system and decompose the end-to-end wall time into model generation, tool execution, orchestration, and other queue/I/O overhead.

All measurements are conducted using the same deployment setting: a 7B policy model on a single H20 GPU with TP=1. For VSP/VSPO, the evaluation additionally uses the Point/Molmo expert worker required by the corresponding tool pipeline. The latency breakdown is shown below:

| Benchmark | Generation | Tool execution | Orchestration | Other / I/O | Total wall time |
| --------- | ---------: | -------------: | ------------: | ----------: | --------------: |
| BLINK-J   |     92.48% |          0.39% |         0.01% |       7.11% |         175.9 s |
| Jigsaw    |     90.95% |          0.55% |         0.02% |       8.49% |       1,275.8 s |
| GUIChat   |     73.85% |         18.65% |         0.01% |       7.50% |       2,744.2 s |
| V*        |     31.16% |         48.11% |         0.05% |      20.69% |       1,411.9 s |
| VSPO      |     43.83% |         54.50% |         0.01% |       1.66% |      13,427.2 s |
| VSP       |     39.11% |         59.42% |         0.01% |       1.47% |       6,541.6 s |

The breakdown reveals two important observations. First, for reasoning-oriented benchmarks such as BLINK-J and Jigsaw, inference cost is dominated by model generation, while tool execution contributes less than 1% of total wall time. Second, for tool-intensive environments such as VSP, VSPO, and V*, tool execution becomes the dominant cost component, reflecting the inherent trade-off between richer external interaction and inference efficiency. Across all benchmarks, orchestration overhead remains negligible, indicating that the majority of latency comes from model computation and external tool execution rather than system coordination.

These results provide a more precise characterization of inference cost beyond CPS. We will add this latency decomposition and clarify the evaluation conditions in the revised manuscript. In addition, we will further discuss the accuracy–latency trade-off under different deployment settings where applicable.

---

## R2-6: Possible image-level leakage in Jigsaw-COCO

**Comment:** 6. Possible image-level leakage in Jigsaw-COCO (published core; verify).

Section C.1 builds training from three patch positions of each image and tests on the fourth patch of the same images. That holds out a patch position, not an image, so the model may have seen most of a test image's content in training. Please confirm and enforce disjointness at the source-image level (image-disjoint splits or COCO-val), with a near-duplicate check. Since this concerns already-published results, at minimum the manuscript should state the guarantee explicitly.

**Response:** 
 We clarify that our training and validation splits are strictly disjoint at the source-image level, rather than only at the patch-position level. Specifically, all patches derived from the same source image are assigned to the same split, ensuring that no source image appears across training and validation.

The released training and validation data are publicly available, allowing this image-level separation to be directly verified. Therefore, the model does not have access to other patches from a validation image during training, and the reported results are not affected by source-image leakage.

We agree that this split guarantee was not stated explicitly enough in the manuscript. We will revise Section C.1 to clearly specify that the split is performed at the source-image level and explicitly describe the image-disjoint evaluation protocol.

---

## R2-7: Validation of the LM judge

**Comment:** 7. The LM-judge is under-validated, including on the new benchmarks.

V*, WebMMU, and GUIQA depend on Qwen2.5-VL-72B as judge. The human check on V* is reported only as "consistent," with no sample size, annotator count, agreement, or blinding. Since tool-augmented answers are more verbose than baselines, a lenient judge may reward length. Provide a quantitative agreement study and a check that verbosity is not advantaged, especially for the new V*/HRBench results.

**Response:** 

To provide a more rigorous evaluation, we conduct a 500-item stratified human semantic audit of the binary judgments produced by Qwen2.5-72B-Instruct.

We choose Qwen2.5-72B-Instruct as the judge model because it provides a reproducible and cost-effective evaluation protocol, enabling large-scale evaluation and repeated analysis. The audit is designed to directly verify whether the judge correctly assesses the semantic correctness of model responses, rather than whether responses are longer or more detailed.

The audit samples GUIChat and WebMMU in proportion to their combined evaluation pool and approximately balances the evaluated models. A fixed random seed (260118631) is used for sampling, and each question is sampled once without replacement. The human auditor independently examines the question, reference answer, and model response under the same binary evaluation criterion used by the judge.

Among the 500 sampled records, 498 are valid. Two WebMMU samples are excluded because they contain only a generic task prefix and an empty reference answer. The audit results are summarized below:

| Audit statistic                   |                       Result |
| --------------------------------- | ---------------------------: |
| Valid / sampled                   |                **498 / 500** |
| Overall agreement                 |         **90.76% (452/498)** |
| Wilson 95% CI                     |            **87.90%–93.00%** |
| Cohen’s κ                         |                    **0.781** |
| Precision / recall / specificity  | **95.59% / 91.29% / 89.44%** |
| False positives / false negatives |                  **15 / 31** |
| GUIChat agreement                 |         **86.29% (170/197)** |
| WebMMU agreement                  |         **93.69% (282/301)** |


The disagreement analysis further shows that the judge is not systematically biased toward tool-augmented responses. Specifically, the number of false negatives (31 cases where the judge rejected human-acceptable answers) is higher than false positives (15 cases where the judge accepted human-incorrect answers), indicating that the judge is not simply favoring more detailed or tool-generated responses.

To further examine whether response verbosity introduces bias, we perform a descriptive analysis by grouping responses into four character-length quartiles. The agreement rates are 87.10%, 91.20%, 93.55%, and 91.20% from the shortest to longest quartile, respectively. Since agreement does not monotonically increase with answer length, these results provide no evidence that longer tool-augmented responses systematically receive higher judge scores. We note that this analysis is descriptive and does not completely rule out all possible verbosity effects.


---


## R2 — Minor concerns

### M1: Moderate “never explicitly trained” and “tools over scale”

**Comment:** Moderate two claims. "Despite never being explicitly trained to do so" sits against an Adaptive Reward (Sec 2.3/A.4) built to regulate tool use and an Adaptive Learning method (Sec 2.4) built for generalization; distinguish not-supervised-at-instance-level from not-trained-at-all. And the "bottleneck shifted from scale to tool quality" claim holds on structured tasks but is contradicted by the paper's own general-task gains (about 4 to 7 points on V*/HRBench/WebMMU vs about 40 on VSP/Jigsaw) and its own "cannot fully offset" caveat; present it as a structured-task finding.

**Response:** We agree. We will replace “despite never being explicitly trained to do so” with **“without instance-level supervision for that evaluation-time interface.”** This distinguishes the absence of a matching demonstration from the broader Adaptive Reward and Adaptive Learning objectives.

We will also restrict “the bottleneck shifts from scale to tool quality” to the studied structured tasks. The revised statement will be: **“On the studied structured visual-reasoning tasks, access to suitable tools can yield larger gains than increasing the base-model scale; on open-ended tasks, tool augmentation provides smaller gains and does not fully offset model capability.”**

### M2: GenReasoner / AdaReasoner naming

**Comment:** Naming. The body says GenReasoner; Figures 1 and 10 and the linked repository say AdaReasoner. In Figures 1 and 10, "AdaReasoner" is plotted beside DeepEyes and PixelReasoner, so a reader may misread the model as a third-party baseline. Use one name throughout, and confirm the released code and checkpoints correspond to this manuscript.

**Response:** We will use **GenReasoner** consistently for the journal model in the title, body, tables, Figures 1 and 10, captions, and checkpoints referenced by the manuscript. The repository README will explicitly map the journal checkpoints to the earlier AdaReasoner code lineage so that readers do not interpret AdaReasoner as an unrelated baseline. We will verify that every linked checkpoint corresponds to the exact configuration reported in the revised manuscript.

### M3: Reproducibility and release

**Comment:** Reproducibility. The data statement covers only public source datasets, not the generated trajectories, exact splits, VSPO and Jigsaw-COCO construction, or seeds. Commit to releasing these, and explain how the 332,649 cold-start samples decompose, since the appendix does not obviously sum to that figure.

**Response:** We will release more than the public source-dataset names. The release package will include:

1. the generated Tool Cold Start trajectories, including reflection/fallback metadata;
2. exact train/validation/test source-image and task-instance manifests;
3. VSPO and Jigsaw-COCO construction scripts;
4. Tool-GRPO prompts, reward implementation, tool schemas, and training hyperparameters;
5. checkpoint identifiers and inference-run metadata, including decoding settings and explicit seeds where available.



### M4: Dedicated limitations

**Comment:** Limitations. A dedicated section is missing (latency and cost, dependence on hand-designed trajectories and external expert models, task-specific tooling). Dependence on tool quality is discussed, but only as an advantage; state it also as a fragility.

**Response:** We will add a dedicated Limitations section covering: (i) increased latency and test-time compute; (ii) dependence on external expert models such as Point/Molmo; (iii) reliance on high-quality, partially hand-designed trajectory blueprints; (iv) task-specific tools and limited evidence for genuinely new capabilities; (v) sensitivity to malformed, missing, contradictory, or slow tool outputs; and (vi) the fact that tool quality is both an advantage and a fragility, because a stronger planner cannot guarantee correctness when its observations are wrong.

### M5: Smaller consistency and terminology issues

**Comment:** Smaller items. VSPO test grids disagree (A.2 lists 5x5/7x7/9x9; C.1 lists 3x3/5x5/7x7/9x9, and A.2 calls 3x3 "larger"). Task count is inconsistent (four tasks in the main text vs "three challenging tasks"). Some table cross-references look off (Table 13 where 11 or 12 seem intended). Terminology varies (Tool-GRPO vs Tool GRPO). Distinguish a syntactically successful tool call from a semantically useful result, since a call can execute and still return misleading information.

**Response:** We will perform a line-by-line consistency pass and make the following corrections:

- use one verified VSPO grid-size specification and remove the erroneous description of 3×3 as “larger”;
- use a single task count throughout the main text and appendix;
- repair the incorrect table cross-references;
- standardize the term **Tool-GRPO**;
- distinguish **syntactic execution success** (the call parses and executes) from **semantic utility/correctness** (the returned observation is relevant and correct);
- report both metrics where available rather than treating execution success as evidence of a useful result.

---

# Closing Response

We sincerely thank the reviewer for the careful and constructive feedback. The comments have helped us substantially improve the clarity, rigor, and positioning of our work. In particular, we have clarified the boundary between this journal extension and the previous AdaReasoner work, refined the scope of our generalization claims, strengthened the validation of experimental reliability through additional analyses, and provided a more comprehensive evaluation of practical considerations including latency, tool failures, and deployment cost.

Following the reviewer’s suggestions, we will revise the manuscript to more clearly distinguish established components from new contributions, avoid overclaiming beyond the evaluated settings, and provide a more transparent discussion of the strengths and limitations of multi-turn, multi-tool agent training.

We appreciate the reviewer’s insightful comments, which have helped us present GenReasoner in a more precise and comprehensive manner.

