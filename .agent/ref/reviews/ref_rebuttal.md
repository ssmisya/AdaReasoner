# PolicyShiftGuard (NeurIPS'26) — Point-by-Point Rebuttal (DRAFT)



## Reviewer EHAh — Rate 4, Confidence 2

We thank the reviewer for identifying schema-bound generalization and benchmark validity as the two concerns most relevant to their assessment. We address both directly, while distinguishing completed evidence from planned checks.

**W1: Limited policy generalization — the Shift split holds out policy *definitions* but keeps the same 7-category ontology, attribute schema, and structured policy format; does the model generalize to genuinely novel policy structures / risk categories / natural policy documents?**

**WR1:** 
We address this concern from 2 aspects of experiments.
**（1）Generalizability to other benchmarks.**
As discussed in Section 4.2 and Table 3, we also evaluate PolicyShiftGuard on two independently constructed benchmarks, achieving **64.1** on UnSafeBench, **61.7** macro-F1 on SafeEditBench, and 69.9 Overall in the cross-benchmark setting. Importantly, both benchmarks adopt independently defined policy settings that are unseen during training. Therefore, the strong performance on both benchmarks demonstrates that PolicyShiftGuard generalizes effectively to previously unseen policy definitions, rather than overfitting to the policies used in our benchmark.



**(2) Generalizability across different policy formats**
To further evaluate whether PolicyShiftGuard generalizes beyond the original policy format and ontology, we conduct two additional evaluations.

First, we remove the A/B/C identifiers from each policy while preserving its semantic content, which tests whether the model relies on superficial policy identifiers rather than understanding the underlying policy requirements. Second, we rephrase each policy using GPT-5.5, modifying the wording and semantic structure while maintaining the original policy intent. This setting further evaluates whether the model can generalize beyond the original policy templates and adapt to diverse policy expressions.

We evaluate PolicyShiftGuard under both settings, and the results are summarized in Table 1. The model maintains consistent performance across the original and modified policy formats, demonstrating that PolicyShiftGuard is not overly dependent on specific identifiers or predefined policy templates, but instead captures the underlying policy semantics.



| Group | Frozen model | Policy-prompt variant | Adaptive/ID Acc. ↑ | Adaptive/ID F1 ↑ | Adaptive/ID PSS ↑ | Adaptive/ID Invalid ↓ | Shift/OOD Acc. ↑ | Shift/OOD F1 ↑ | Shift/OOD PSS ↑ | Shift/OOD Invalid ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours** | **PolicyShiftGuard-3B** | Canonical policy | 86.6 | 85.6 | 79.0 | 0.0 | 61.1 | 47.8 | 50.5 | 0.0 |
| | | Without A/B/C/... identifiers | 86.0 | 85.1 | 75.6 | 0.0 | 60.2 | 46.8 | 50.5 | 0.0 |
| | | Rephrased and identifier-free | 87.0 | 86.6 | 80.4 | 0.0 | 60.5 | 51.3 | 50.0 | 0.0 |
| | **PolicyShiftGuard-7B** | Canonical policy | 86.3 | 86.8 | 73.8 | 0.0 | 69.9 | 67.0 | 70.4 | 0.0 |
| | | Without A/B/C/... identifiers | 86.1 | 86.6 | 73.2 | 0.0 | 69.6 | 66.3 | 69.9 | 0.0 |
| | | Rephrased and identifier-free | 81.5 | 83.0 | 70.8 | 0.0 | 67.3 | 63.4 | 64.5 | 0.0 |
| **Baselines** | Qwen2.5-VL-7B | Canonical policy | 54.5 | 29.0 | 9.5 | 0.0 | 50.7 | 12.1 | 0.0 | 0.0 |
| | | Without A/B/C/... identifiers | 53.3 | 28.3 | 6.0 | 0.0 | 50.5 | 12.4 | 1.1 | 0.0 |
| | | Rephrased and identifier-free | 55.2 | 27.7 | 13.7 | 0.0 | 50.4 | 10.5 | 1.6 | 0.0 |
| | QwenGuard-7B | Canonical policy | 44.6 | 39.0 | 10.2 | 50.9 | 49.9 | 39.5 | 9.1 | 41.3 |
| | | Without A/B/C/... identifiers | 40.4 | 34.9 | 12.7 | 38.1 | 53.3 | 48.7 | 17.0 | 39.2 |
| | | Rephrased and identifier-free | 42.6 | 40.3 | 13.9 | 36.6 | 49.1 | 44.4 | 16.0 | 40.9 |
**Table 1. Robustness of PolicyShiftGuard under different policy formats.**



****

**W2: [Training–Evaluation Coupling] BP-Adapt is explicitly trained on same-image pass/block policy pairs, while PSS evaluates the same type of paired policy flip. This close alignment may overestimate broader policy-reasoning ability. Additional evaluation under different policy templates, formats, or independently constructed policy-shift benchmarks would strengthen the conclusions.**

**WR2:** 
We clarify this concern from two perspectives: the design alignment between BP-Adapt and PSS, and the generalizability of PolicyShiftGuard beyond the original evaluation setting.

**(1) Clarification of the coupling between BP-Adapt and PSS.**
The coupling between BP-Adapt and PSS is intentionally designed to directly reflect our core motivation. Specifically, our goal is to enable the guardrail model to make accurate, policy-aware decisions when the same image may receive different labels under different policies, while satisfying strict latency constraints. Accordingly, this objective is both explicit and measurable, allowing it to be directly optimized during training and systematically evaluated at test time. As a result, our method and evaluation are tightly aligned with the central motivation of the paper, ensuring that the proposed framework is optimized and assessed against the exact capability it is designed to achieve.

**(2) Additional evaluation for generalizability**
We also evaluate the generalizability of PolicyShiftGuard on other benchmarks. As discussed in Section 4.2 and Table 3, we also evaluate PolicyShiftGuard on two independently constructed benchmarks, achieving **64.1** on UnSafeBench, **61.7** macro-F1 on SafeEditBench, and 69.9 Overall in the cross-benchmark setting. Importantly, both benchmarks adopt independently defined policy settings that are unseen during training. Therefore, the strong performance on both benchmarks demonstrates that PolicyShiftGuard generalizes effectively to previously unseen policy definitions, rather than overfitting to the policies used in our benchmark.


Moreover, as discussed in WR1, we further evaluate PolicyShiftGuard under modified policy formats, including identifier removal and policy rephrasing. We evaluate PolicyShiftGuard under both settings, and the results are summarized in Table 1. The model maintains consistent performance across the original and modified policy formats, demonstrating that PolicyShiftGuard is not overly dependent on specific identifiers or predefined policy templates, but instead captures the underlying policy semantics and generalizes across diverse policy expressions.

Overall, these results demonstrate that the alignment between BP-Adapt and PSS is not a limitation of the evaluation design, but a deliberate choice to optimize and measure the intended policy-conditioned decision capability. Meanwhile, evaluations beyond the original benchmark setting further support that PolicyShiftGuard generalizes beyond specific policies and templates.
****


**W3.1: Benchmark validity — attribute labels are VLM-derived and rule-based; human agreement does not verify attribute correctness. Attribute-level human validation is needed.**

**WR3.1:** 
We perform a fine-grained human audit at the attribute level to directly validate the VLM-derived attributes used for deterministic policy rules, rather than only evaluating the final policy labels.

Specifically, we manually verify 1,060 attribute-level annotations for Adaptive and 1,060 attribute-level annotations for Shift (53 attributes × 20 images per split). Human reviewers are blinded to VLM votes, rule-derived labels, and model predictions. The sampled images are stratified to include non-unanimous cases and rare/high-risk categories (e.g., child safety, weapons, and PII), ensuring that the validation is not dominated by easy negative examples.

The results are summarized below. We report both the agreement between VLM-derived attributes and human verification, as well as the consistency of the VLM voting process. The high Human–VLM agreement (98.49% for Adaptive and 98.68% for Shift) demonstrates that the generated attributes are strongly aligned with human judgments, providing direct evidence for the reliability of the attribute-based labeling pipeline 

| Split | Images | Attribute-level judgments | Attribute coverage | Inter-VLM vote agreement | Human-VLM agreement |
|---|---:|---:|---|---|---|
| Adaptive | 20 | 1,060 | 53/53 attributes | 95.85% (1,016/1,060) | 98.49% |
| Shift | 20 | 1,060 | 53/53 attributes | 95.28% (1,010/1,060) | 98.68% |

**Table 2. Attribute-level human validation of VLM-derived annotations.**




**W3.2: Moreover, the 2,000 evaluation instances contain only 265 unique images, and the Adaptive and Shift splits share 17 images. Results on fully image-disjoint subsets are needed.**

**WR3.2:** 

**(1) Regarding the 17 overlapping images.** 
Although the Adaptive and Shift splits share 17 images, both splits are strictly separated from the training set, ensuring that no image-level data leakage occurs. Therefore, the evaluation on each subset remains independent with respect to the training data, and the comparison is fair under the intended benchmark setting. 

Nevertheless, to further address this concern and provide a more rigorous evaluation, we additionally conduct experiments by removing all overlapping images between the two test splits. As shown in Table 3, the performance remains consistent, demonstrating that our conclusions are not affected by the shared images.



| Split (7B) | Acc | F1 | PSS |
|---|---|---|---|
| Shift — full (152 img) | 69.9 | 67.0 | 70.4 |
| **Shift — 17 shared images removed (135 img)** | **70.4** | **68.4** | **69.9** |
| Adaptive — full (130 img) | 86.3 | 86.8 | 73.8 |
| **Adaptive — 17 shared images removed (113 img)** | **87.6** | **88.1** | **79.6** |

**Table 3: Results of PolicyShiftGuard-7B on PolicyShiftBench with overlapping images removed.**

**(2) Regarding the scale** 
We clarify that the relatively small number of unique images is an intentional design choice motivated by two considerations.

**First, it directly follows from the core objective of PolicyShiftBench**: evaluating whether a model can make different decisions for the same image when the underlying policy changes. Unlike conventional safety benchmarks that primarily emphasize image diversity, our benchmark requires multiple policy-conditioned instances for each visual example to assess whether the model can adapt its decisions according to policy changes. Therefore, a higher policy-to-image ratio is essential for measuring the target capability.

**Second, constructing such a benchmark inherently requires more challenging data collection.** Specifically, we need carefully curated policy-sensitive examples where images lie near policy boundaries and can legitimately receive different labels under different policies. Such ambiguous cases are substantially harder to identify and collect than standard safety examples, which naturally limits the available image pool.


Moreover, to further mitigate the concern regarding image scale, we construct **PolicyShiftBench-PLUS** following the same policy-conditioned data construction protocol. Specifically, PolicyShiftBench-PLUS is constructed from an expanded pool of 11,000 candidate images using the same VLM-assisted annotation, rule-based generation, deduplication, and policy-discriminative sampling procedures. The resulting benchmark contains 2,016 policy-conditioned instances from 748 unique images (359 images from Adaptive and 389 images from Shift), with an exactly balanced label distribution of 1:1 between block and pass. We re-evaluate PolicyShiftGuard on this larger-scale benchmark, and the results are summarized in the table below.

| Model | A Acc | A F1 | A PSS | S Acc | S F1 | S PSS | Avg Acc | Avg F1 | Avg PSS | Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **General-purpose MLLMs** |  |  |  |  |  |  |  |  |  |  |
| Qwen3.5-0.8B | 50.3 | 2.7 | 2.9 | 49.8 | 0.8 | 0.0 | 50.0 | 1.8 | 1.4 | 63.6 |
| Qwen3.5-2B | 63.4 | 0.0 | 0.0 | 55.1 | 3.3 | 0.0 | 59.2 | 1.7 | 0.0 | 93.1 |
| Qwen3.5-4B | 64.6 | 60.0 | 4.9 | 54.0 | 53.8 | 0.0 | 59.3 | 56.9 | 2.4 | 136.4 |
| Qwen3.5-35B-A3B | 67.3 | 62.8 | 29.0 | 62.0 | 50.8 | 10.0 | 64.6 | 56.8 | 19.5 | 198.7 |
| Qwen3.5-4B (Think) | 80.2 | 59.1 | 45.5 | 73.4 | 52.0 | 0.0 | 76.8 | 55.5 | 22.7 | 18204.6 |
| Qwen2.5-VL-3B | 54.8 | 61.6 | 7.6 | 49.5 | 59.2 | 2.0 | 52.1 | 60.4 | 4.8 | 80.6 |
| Qwen2.5-VL-7B | 53.4 | 19.8 | 0.0 | 55.1 | 21.5 | 0.0 | 54.2 | 20.6 | 0.0 | 107.8 |
| **Specialized Guardrails** |  |  |  |  |  |  |  |  |  |  |
| Llama Guard-4-12B | 52.5 | 13.1 | 2.2 | 52.0 | 11.4 | 0.0 | 52.2 | 12.2 | 1.1 | 1116.2 |
| GuardReasoner-VL-3B | 56.2 | 60.1 | 1.2 | 48.3 | 57.1 | 0.0 | 52.3 | 58.6 | 0.6 | 7915.7 |
| GuardReasoner-VL-7B | 57.4 | 56.2 | 5.8 | 47.2 | 53.8 | 12.0 | 52.3 | 55.0 | 8.9 | 5668.4 |
| SafeGuard-VL-RL-7B | 57.3 | 46.7 | 1.4 | 57.3 | 43.9 | 10.0 | 57.3 | 45.3 | 5.7 | 108.8 |
| **Ours** |  |  |  |  |  |  |  |  |  |  |
| **PolicyShiftGuard-3B** | 81.1 | 80.4 | 68.8 | 59.0 | 53.8 | 32.0 | 70.0 | 67.1 | 50.4 | 71.8 |
| **PolicyShiftGuard-7B** | 79.3 | 81.4 | 62.3 | 73.7 | 74.3 | 50.0 | 76.5 | 77.9 | 56.2 | 96.0 |

**Table 4. Main results on PolicyShiftBench-PLUS.**





**E1： The dataset contains sensitive visual content, PII-related examples, and potentially copyrighted images. The authors should clarify image provenance, redistribution rights, consent where applicable, annotator protection, and whether the internal human audit required institutional approval or exemption.**


**ER1:**

We clarify that PolicyShiftBench was constructed by aggregating existing publicly available datasets and carefully curated category-specific image collections, with the goal of evaluating policy-conditioned visual safety decisions rather than redistributing sensitive content.

**Image provenance:** Our image pool is constructed from the following publicly available sources: UnsafeBench, LLaVA-Guard, VisionHarm-500K, NeuralShell Gore-Blood, an AdImageNet-derived subset. We also incorporated safe-control images from DiffusionDB, ShareGPT-4o-Image, and BLIP3o Pretrain Long Caption. Among the retained samples.

**Rights, consent, and annotation protection:** Since our benchmark is derived from existing datasets, we follow the original dataset licenses and usage policies rather than collecting or redistributing newly scraped user data. 

**Human Audit**: The human audit only involved internal inspection of existing benchmark samples and did not involve human subjects, user studies, or collection of personal data. Therefore, institutional review was not required. 

Overall, we acknowledge the importance of responsible dataset construction and have clarified the provenance and safeguards of PolicyShiftBench. We will further document the source information, usage constraints, and ethical considerations in the revised manuscript.


****

**L1:** The authors acknowledge limitations regarding modality, language, and the finite policy catalog, but the discussion is incomplete. It should also address the close coupling between BP-Adapt and PSS, dependence on VLM-derived attributes and deterministic rules, the limited number of unique evaluation images, and generalization beyond the shared policy ontology and templates.
 
**LR1:** 
We thank the reviewer for pointing out these important limitations. We agree that it is important to clearly distinguish between the capability directly evaluated by PolicyShiftBench and broader, unrestricted policy reasoning. We address each concern below.

**(1) Coupling between BP-Adapt and PSS.**
As discussed in WR2, the coupling between BP-Adapt and PSS is intentional and reflects the core objective of our work: enabling a guardrail model to make policy-aware decisions when the same image requires different judgments under different policies. Therefore, optimizing BP-Adapt toward this capability and evaluating it with PSS are aligned by design rather than a methodological limitation. However, we agree that this construct alignment alone does not establish perfect general policy reasoning beyond the evaluated setting. We will clarify this scope in the revised manuscript and discuss potential extensions toward broader policy generalization.


**(2) Limited unique-image coverage.**
As discussed in WR3.2, the relatively high instance-to-image ratio is also an intentional design choice. Unlike conventional image-level generalization benchmarks, PolicyShiftBench aims to evaluate whether models can adapt their decisions for the same visual content under different policy specifications. Therefore, multiple policy-conditioned instances per image are essential for measuring this capability. Nevertheless, to further address concerns regarding image diversity, we additionally construct a larger evaluation set with more unique images and report the corresponding results, which remain consistent with our original findings.

**(3) Generalization beyond the shared policy ontology and templates.**
We agree that the original Shift split mainly evaluates generalization beyond seen policy instances while keeping the same underlying ontology and schema. To better assess broader generalization, as discussed in WR1, we provide complementary evidence from two directions: (i) evaluation on independently constructed external benchmarks (UnSafeBench and SafeEditBench), where policies and data distributions differ from those used in training; and (ii) policy rephrasing experiments that modify policy expressions while preserving their semantics. These results demonstrate that PolicyShiftGuard is not solely relying on the original policy templates or identifiers. At the same time, we acknowledge that unrestricted generalization to arbitrary natural-language policies remains an open challenge and will be added as a limitation.

**(4) Dependence on VLM-derived attributes and deterministic rules.**
We agree that the current benchmark construction relies on VLM-derived attributes and deterministic rules, mainly due to the scalability and cost considerations of large-scale policy-conditioned annotation. To mitigate this concern, as discussed in WR3.1, we conducted additional human validation and calibration of attribute-level labels, including blind annotation and adjudication. Nevertheless, we acknowledge that this construction pipeline introduces a dependency on the quality of intermediate attributes and rule design. We will explicitly include this dependency as a limitation and discuss future directions toward more fully human-verified or end-to-end annotated benchmarks.

Overall, we appreciate the reviewer’s suggestions. We will revise the manuscript to more clearly state the evaluated capability, external validity boundaries, and remaining challenges of PolicyShiftBench.

---

## Reviewer 55PB — Rate 3, Confidence 4

We thank the reviewer for the rigorous and constructive review. We appreciate the insightful comments on PSS, latency analysis, and ablation studies. We address these concerns with additional analyses and experiments, and clarify the scope and limitations of our claims.

**W1: The image base may be small for the scope claimed (Lines 102–104, Table 7). The benchmark has 265 unique images across 28 policy variants. 
The Shift Split has 152 images across 12 held-out policies. Most policy-category combinations are represented by very few images. The authors should report per-policy-variant image counts and acknowledge that per-category PSS values in Figure 4 carry substantial variance at this scale. The generalisation claims to held-out policies are not well-supported by 12 policies over 152 images.**

**WR1:** We clarify that the relatively small number of unique images is an intentional design choice motivated by two considerations.

**First, it directly follows from the core objective of PolicyShiftBench**: evaluating whether a model can make different decisions for the same image when the underlying policy changes. Unlike conventional safety benchmarks that primarily emphasize image diversity, our benchmark requires multiple policy-conditioned instances for each visual example to assess whether the model can adapt its decisions according to policy changes. Therefore, a higher policy-to-image ratio is essential for measuring the target capability.

**Second, constructing such a benchmark inherently requires more challenging data collection.** Specifically, we need carefully curated policy-sensitive examples where images lie near policy boundaries and can legitimately receive different labels under different policies. Such ambiguous cases are substantially harder to identify and collect than standard safety examples, which naturally limits the available image pool.

To further address the concern regarding generalization beyond the current benchmark setting, we conduct three additional evaluations.


**(1) Generalization to independently constructed benchmarks.**
As discussed in Section 4.2 and Table 3, we evaluate PolicyShiftGuard on two independently constructed benchmarks, achieving 64.1 on UnSafeBench, 61.7 macro-F1 on SafeEditBench, and 69.9 Overall in the cross-benchmark setting. Both benchmarks contain independently defined policies and data distributions unseen during training. These results provide evidence that PolicyShiftGuard generalizes beyond the specific policies and images in PolicyShiftBench.

**(2) Generalization across different policy formats.**
We further evaluate PolicyShiftGuard under two policy-format variations: (i) removing A/B/C identifiers while preserving policy semantics, and (ii) rephrasing policies with different linguistic structures while maintaining the original intent. As shown in the table below, although these transformations introduce additional challenges, PolicyShiftGuard maintains strong performance and consistently outperforms baselines, demonstrating robustness beyond specific policy templates.


| Group | Frozen model | Policy-prompt variant | Adaptive/ID Acc. ↑ | Adaptive/ID F1 ↑ | Adaptive/ID PSS ↑ | Adaptive/ID Invalid ↓ | Shift/OOD Acc. ↑ | Shift/OOD F1 ↑ | Shift/OOD PSS ↑ | Shift/OOD Invalid ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Ours** | **PolicyShiftGuard-3B** | Canonical policy | 86.6 | 85.6 | 79.0 | 0.0 | 61.1 | 47.8 | 50.5 | 0.0 |
| | | Without A/B/C/... identifiers | 86.0 | 85.1 | 75.6 | 0.0 | 60.2 | 46.8 | 50.5 | 0.0 |
| | | Rephrased and identifier-free | 87.0 | 86.6 | 80.4 | 0.0 | 60.5 | 51.3 | 50.0 | 0.0 |
| | **PolicyShiftGuard-7B** | Canonical policy | 86.3 | 86.8 | 73.8 | 0.0 | 69.9 | 67.0 | 70.4 | 0.0 |
| | | Without A/B/C/... identifiers | 86.1 | 86.6 | 73.2 | 0.0 | 69.6 | 66.3 | 69.9 | 0.0 |
| | | Rephrased and identifier-free | 81.5 | 83.0 | 70.8 | 0.0 | 67.3 | 63.4 | 64.5 | 0.0 |
| **Baselines** | Qwen2.5-VL-7B | Canonical policy | 54.5 | 29.0 | 9.5 | 0.0 | 50.7 | 12.1 | 0.0 | 0.0 |
| | | Without A/B/C/... identifiers | 53.3 | 28.3 | 6.0 | 0.0 | 50.5 | 12.4 | 1.1 | 0.0 |
| | | Rephrased and identifier-free | 55.2 | 27.7 | 13.7 | 0.0 | 50.4 | 10.5 | 1.6 | 0.0 |
| | QwenGuard-7B | Canonical policy | 44.6 | 39.0 | 10.2 | 50.9 | 49.9 | 39.5 | 9.1 | 41.3 |
| | | Without A/B/C/... identifiers | 40.4 | 34.9 | 12.7 | 38.1 | 53.3 | 48.7 | 17.0 | 39.2 |
| | | Rephrased and identifier-free | 42.6 | 40.3 | 13.9 | 36.6 | 49.1 | 44.4 | 16.0 | 40.9 |

**(3) Evaluation on PolicyShiftGuard-PLUS.**
Finally, to further mitigate the concern regarding image scale, we construct **PolicyShiftBench-PLUS** following the same policy-conditioned data construction protocol. Specifically, PolicyShiftBench-PLUS is constructed from an expanded pool of 11,000 candidate images using the same VLM-assisted annotation, rule-based generation, deduplication, and policy-discriminative sampling procedures. The resulting benchmark contains 2,016 policy-conditioned instances from 748 unique images (359 images from Adaptive and 389 images from Shift), with an exactly balanced label distribution of 1:1 between block and pass. We re-evaluate PolicyShiftGuard on this larger-scale benchmark, and the results are summarized in the table below.

| Model | A Acc | A F1 | A PSS | S Acc | S F1 | S PSS | Avg Acc | Avg F1 | Avg PSS | Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **General-purpose MLLMs** |  |  |  |  |  |  |  |  |  |  |
| Qwen3.5-0.8B | 50.3 | 2.7 | 2.9 | 49.8 | 0.8 | 0.0 | 50.0 | 1.8 | 1.4 | 63.6 |
| Qwen3.5-2B | 63.4 | 0.0 | 0.0 | 55.1 | 3.3 | 0.0 | 59.2 | 1.7 | 0.0 | 93.1 |
| Qwen3.5-4B | 64.6 | 60.0 | 4.9 | 54.0 | 53.8 | 0.0 | 59.3 | 56.9 | 2.4 | 136.4 |
| Qwen3.5-35B-A3B | 67.3 | 62.8 | 29.0 | 62.0 | 50.8 | 10.0 | 64.6 | 56.8 | 19.5 | 198.7 |
| Qwen3.5-4B (Think) | 80.2 | 59.1 | 45.5 | 73.4 | 52.0 | 0.0 | 76.8 | 55.5 | 22.7 | 18204.6 |
| Qwen2.5-VL-3B | 54.8 | 61.6 | 7.6 | 49.5 | 59.2 | 2.0 | 52.1 | 60.4 | 4.8 | 80.6 |
| Qwen2.5-VL-7B | 53.4 | 19.8 | 0.0 | 55.1 | 21.5 | 0.0 | 54.2 | 20.6 | 0.0 | 107.8 |
| **Specialized Guardrails** |  |  |  |  |  |  |  |  |  |  |
| Llama Guard-4-12B | 52.5 | 13.1 | 2.2 | 52.0 | 11.4 | 0.0 | 52.2 | 12.2 | 1.1 | 1116.2 |
| GuardReasoner-VL-3B | 56.2 | 60.1 | 1.2 | 48.3 | 57.1 | 0.0 | 52.3 | 58.6 | 0.6 | 7915.7 |
| GuardReasoner-VL-7B | 57.4 | 56.2 | 5.8 | 47.2 | 53.8 | 12.0 | 52.3 | 55.0 | 8.9 | 5668.4 |
| SafeGuard-VL-RL-7B | 57.3 | 46.7 | 1.4 | 57.3 | 43.9 | 10.0 | 57.3 | 45.3 | 5.7 | 108.8 |
| **Ours** |  |  |  |  |  |  |  |  |  |  |
| **PolicyShiftGuard-3B** | 81.1 | 80.4 | 68.8 | 59.0 | 53.8 | 32.0 | 70.0 | 67.1 | 50.4 | 71.8 |
| **PolicyShiftGuard-7B** | 79.3 | 81.4 | 62.3 | 73.7 | 74.3 | 50.0 | 76.5 | 77.9 | 56.2 | 96.0 |




****

**W2: Attribute annotation relies on three VLMs with no independent validity check; 97.5% unanimity is agreement between correlated models; the human audit measures label quality after rules, not attribute-level accuracy; labels are only as reliable as unvalidated VLM attributes.**

**WR2:** Beyond measuring VLM consensus, we conduct a fine-grained human audit directly on the VLM-derived attributes to independently validate the attribute-level annotations used by the deterministic rules.

Specifically, we manually verify 1,060 attribute-level annotations for Adaptive and another 1,060 attribute-level annotations for Shift (53 attributes × 20 images per split). Human reviewers are blinded to the VLM votes, rule-derived labels, and model predictions, ensuring that the verification is independent from the original annotation pipeline. The sampled images are stratified to include non-unanimous cases and rare/high-risk categories (e.g., child safety, weapons, and PII), rather than being dominated by easy negative examples.

The results are summarized below. We report both the agreement between VLM-derived attributes and human verification and the consistency of the VLM voting process. The high Human–VLM agreement (98.49% for Adaptive and 98.68% for Shift) provides direct evidence that the generated attributes are strongly aligned with human judgments, addressing the concern that benchmark labels may rely on unvalidated VLM attributes.

| Split | Images | Attribute-level judgments | Attribute coverage | Inter-VLM vote agreement | Human-VLM agreement |
|---|---:|---:|---|---|---|
| Adaptive | 20 | 1,060 | 53/53 attributes | 95.85% (1,016/1,060) | 98.49% |
| Shift | 20 | 1,060 | 53/53 attributes | 95.28% (1,010/1,060) | 98.68% |

****

**W3: Privacy/PII on Shift collapses (Table 13: 7B = 15.3% accuracy), below all closed-source baselines and far below human, and is not discussed.**

**WR3:** 

We thank the reviewer for highlighting this failure case. After further analysis, we find that the low performance on the Shift privacy/PII category is primarily due to the highly challenging out-of-distribution policy design in this subset, rather than a general inability of PolicyShiftGuard to handle privacy-related content.

Specifically, the Shift split for the PII category evaluates a single held-out policy variant:

Policy C: Sensitive Data Isolation Mode (Secure Data Entry / OCR)

This policy follows an inverse allow-list logic: it only permits document-related inputs (e.g., ID cards and credit cards) and blocks all other non-document images (e.g., selfies, landscapes, and general lifestyle images) as invalid inputs or privacy leakage risks. Unlike conventional privacy policies that focus on identifying the presence of sensitive information, this policy requires the model to follow a highly specific purpose-driven constraint.

Therefore, this setting represents a particularly challenging OOD policy-shift case, where the desired decision boundary is strict and counter-intuitive. The performance degradation in this category reflects the difficulty of adapting to such an uncommon policy specification, rather than a failure to recognize privacy-related content itself. Notably, this challenging behavior is also observed across different models, further indicating that the difficulty mainly arises from the policy formulation.

We will clarify this challenging case in the revised manuscript and discuss it as an important direction for future evaluation of more diverse and complex privacy policies.


****

**Q1: Bibliography is almost entirely 2025/2026 arXiv preprints; SafeEditBench, QwenGuard, SafeGuard-VL-RL appear to be unreviewed concurrent submissions. Are any compared systems peer-reviewed?**

**A1:** Yes. Our source-level audit identifies peer-reviewed comparisons including GuardReasoner-VL (NeurIPS 2025), LlavaGuard and its QwenGuard-7B variant (ICML 2025), UnSafeBench (ACM CCS 2025), MM-SafetyBench (ECCV 2024), VSCBench (Findings of ACL 2025), and FigStep (AAAI 2025). SafeEditBench and SafeGuard-VL-RL come from the same paper, now published at CVPR 2026; because it is concurrent with our 2026 work/submission cycle, we will explicitly label it “CVPR 2026; concurrent work” (or “concurrent at submission” after confirming the exact dates). The audit also found an attribution error in our current bibliography: QwenGuard was introduced in the ICML 2025 LlavaGuard paper, not in the SafeGuard-VL/SafeEditBench paper, and we will correct this. Other rows are public technical-report/model releases (Qwen2.5-VL, Qwen3.5, Llama Guard 4, and ShieldGemma 2) or closed APIs (Claude-Sonnet-4.6, GPT-5.4, and Gemini-3-Flash-Preview), rather than peer-reviewed method papers. We will add a venue/status column, cite the exact evaluated versions, and avoid implying independent peer review for those categories.



****

**Q2: Claude-Sonnet-4.6 and GPT-5.4 have no technical reports — what prompting format and policy-conditioning template were used?**

**A2:** We clarify that Claude-Sonnet-4.6 and GPT-5.4 are evaluated through their standard API interfaces, following the official API usage recommendations. During inference, we use the same dataset-native policy question and output format as other evaluated models, ensuring that the policy-conditioning mechanism is consistent across models.

Specifically, each API call contains a single user message including the policy-conditioned query, the common output-format instruction, and the image input. Other inference settings (e.g., output format, retry strategy, timeout handling, and evaluation parser) are kept consistent with the settings used for other models. For GPT-5.4, we follow the API configuration recommended in the official documentation, while avoiding provider-specific modifications that could introduce additional advantages.

The detailed evaluation settings are summarized below:
| Item                | Setting                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------- |
| Model access        | Standard API calls for each closed-source model                                             |
| Policy conditioning | Dataset-native policy question + common output instruction                                  |
| Image input         | JPEG image input through API request                                                        |
| Output format       | Unified true/false line format                                                              |
| Evaluation parser   | Shared `parse_response` function                                                            |
| Decoding settings   | Aligned with the default API configuration and consistent evaluation protocol across models |
| Retry / timeout     | 5 retries with exponential backoff; 120-second timeout                                      |


****

**Q3: Adaptive and Shift share 17 images — what prevents label leakage between the in-distribution and held-out tracks?**

**A3:** 
We clarify that the 17 shared images between Adaptive and Shift do not introduce label leakage from the training set, as both evaluation splits are strictly separated from all training images. Moreover, PolicyShiftBench is designed to evaluate policy-conditioned decision making, where the same image can intentionally appear under different policies to test whether the model adapts its judgment according to policy changes, rather than relying solely on image memorization.

Nevertheless, we agree that overlap between evaluation splits may raise concerns regarding the independence of the reported comparisons. To further address this issue, we conduct an additional evaluation by removing all shared images between Adaptive and Shift and re-evaluate PolicyShiftGuard-7B on the resulting image-disjoint subsets.

As shown in the table below, the performance remains consistent after removing the 17 overlapping images. Specifically, Shift changes from 69.9/67.0/70.4 to 70.4/68.4/69.9 in Acc/F1/PSS, while Adaptive changes from 86.3/86.8/73.8 to 87.6/88.1/79.6. These results demonstrate that our conclusions are not driven by the shared images, and the reported generalization behavior remains robust under a stricter image-disjoint evaluation setting.

| Split (7B)                                        | Acc      | F1       | PSS      |
| ------------------------------------------------- | -------- | -------- | -------- |
| Shift — full (152 img)                            | 69.9     | 67.0     | 70.4     |
| **Shift — 17 shared images removed (135 img)**    | **70.4** | **68.4** | **69.9** |
| Adaptive — full (130 img)                         | 86.3     | 86.8     | 73.8     |
| **Adaptive — 17 shared images removed (113 img)** | **87.6** | **88.1** | **79.6** |

---

Overall, we appreciate the reviewer’s valuable suggestions. We will revise the manuscript to more clearly state the scope of our claims, strengthen the evaluation of generalization and robustness, and clarify the efficiency advantages of PolicyShiftGuard.

---

## Reviewer tKKf — Rate 3, Confidence 3

We thank the reviewer for the thoughtful and constructive feedback. In the following, we address each concern point-by-point and provide additional analyses and experiments where appropriate.

**W1: The data recipe relies on majority voting of VLMs, which may not be reliable for rare yet important cases.**

**WR1:** 
We clarify that our benchmark construction includes both policy-label-level human auditing and attribute-level validation to ensure the reliability of the generated labels.

First, as discussed in Section 2.3, we perform a blind human evaluation to directly assess the quality of the final policy labels. Human auditing is separated into a blind model-evaluation task and a data-quality verification task. The paper-facing audit reports 88% and 90% blind human accuracy on the Adaptive and Shift splits, respectively, with a 95% qualified rate for both splits, demonstrating that the final benchmark labels are reliable.

Second, to further examine whether majority voting over VLM-derived attributes may introduce errors, we conduct a fine-grained human audit at the attribute level. Specifically, we manually verify 1,060 attribute-level annotations for Adaptive and 1,060 attribute-level annotations for Shift (53 attributes × 20 images per split). Human reviewers are blinded to VLM votes, rule-derived labels, and model predictions. The sampled images are stratified to include non-unanimous cases and rare/high-risk categories (e.g., child safety, weapons, and PII), rather than being dominated by easy negative examples.

The results are summarized below. The high Human–VLM agreement (98.49% for Adaptive and 98.68% for Shift) demonstrates that the VLM-derived attributes are strongly aligned with human judgments, providing direct evidence that the majority-voting pipeline remains reliable even for challenging cases.

| Split | Images | Attribute-level judgments | Attribute coverage | Inter-VLM vote agreement | Human-VLM agreement |
|---|---:|---:|---|---|---|
| Adaptive | 20 | 1,060 | 53/53 attributes | 95.85% (1,016/1,060) | 98.49% |
| Shift | 20 | 1,060 | 53/53 attributes | 95.28% (1,010/1,060) | 98.68% |



****

**W2.1 / Limitations: Seven risk categories is a very large scope; 265 images may not be enough**

**WR2.1:** 
We clarify that the relatively small number of unique images is an intentional design choice motivated by two considerations.

**(1) The benchmark prioritizes policy variation diversity over image diversity.**
Specifically, our goal is to test whether a model can produce different judgments for the same image when the underlying policy changes. Therefore, the critical requirement is to construct sufficient policy-conditioned instances per image, rather than simply maximizing the number of unique images. A higher instance-to-image ratio is essential for evaluating whether the model truly adapts its decisions according to policy changes.

**(2) Policy-sensitive examples are inherently difficult to collect.**
Unlike conventional safety benchmarks that can be expanded by collecting more diverse images, PolicyShiftBench requires carefully curated examples where the same image lies near policy boundaries and can legitimately receive different labels under different policies. Such ambiguous cases are substantially harder to identify and collect, which naturally limits the available image pool.

Nevertheless, to further address the reviewer’s concern regarding benchmark scale, we construct **PolicyShiftBench-PLUS** following the same policy-conditioned data construction protocol. Specifically, PolicyShiftBench-PLUS is constructed from an expanded pool of 11,000 candidate images using the same VLM-assisted annotation, rule-based generation, deduplication, and policy-discriminative sampling procedures. The resulting benchmark contains 2,016 policy-conditioned instances from 748 unique images (359 images from Adaptive and 389 images from Shift), with an exactly balanced label distribution of 1:1 between block and pass. We re-evaluate PolicyShiftGuard on this larger-scale benchmark, and the results are summarized in the table below.

| Model | A Acc | A F1 | A PSS | S Acc | S F1 | S PSS | Avg Acc | Avg F1 | Avg PSS | Time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **General-purpose MLLMs** |  |  |  |  |  |  |  |  |  |  |
| Qwen3.5-0.8B | 50.3 | 2.7 | 2.9 | 49.8 | 0.8 | 0.0 | 50.0 | 1.8 | 1.4 | 63.6 |
| Qwen3.5-2B | 63.4 | 0.0 | 0.0 | 55.1 | 3.3 | 0.0 | 59.2 | 1.7 | 0.0 | 93.1 |
| Qwen3.5-4B | 64.6 | 60.0 | 4.9 | 54.0 | 53.8 | 0.0 | 59.3 | 56.9 | 2.4 | 136.4 |
| Qwen3.5-35B-A3B | 67.3 | 62.8 | 29.0 | 62.0 | 50.8 | 10.0 | 64.6 | 56.8 | 19.5 | 198.7 |
| Qwen3.5-4B (Think) | 80.2 | 59.1 | 45.5 | 73.4 | 52.0 | 0.0 | 76.8 | 55.5 | 22.7 | 18204.6 |
| Qwen2.5-VL-3B | 54.8 | 61.6 | 7.6 | 49.5 | 59.2 | 2.0 | 52.1 | 60.4 | 4.8 | 80.6 |
| Qwen2.5-VL-7B | 53.4 | 19.8 | 0.0 | 55.1 | 21.5 | 0.0 | 54.2 | 20.6 | 0.0 | 107.8 |
| **Specialized Guardrails** |  |  |  |  |  |  |  |  |  |  |
| Llama Guard-4-12B | 52.5 | 13.1 | 2.2 | 52.0 | 11.4 | 0.0 | 52.2 | 12.2 | 1.1 | 1116.2 |
| GuardReasoner-VL-3B | 56.2 | 60.1 | 1.2 | 48.3 | 57.1 | 0.0 | 52.3 | 58.6 | 0.6 | 7915.7 |
| GuardReasoner-VL-7B | 57.4 | 56.2 | 5.8 | 47.2 | 53.8 | 12.0 | 52.3 | 55.0 | 8.9 | 5668.4 |
| SafeGuard-VL-RL-7B | 57.3 | 46.7 | 1.4 | 57.3 | 43.9 | 10.0 | 57.3 | 45.3 | 5.7 | 108.8 |
| **Ours** |  |  |  |  |  |  |  |  |  |  |
| **PolicyShiftGuard-3B** | 81.1 | 80.4 | 68.8 | 59.0 | 53.8 | 32.0 | 70.0 | 67.1 | 50.4 | 71.8 |
| **PolicyShiftGuard-7B** | 79.3 | 81.4 | 62.3 | 73.7 | 74.3 | 50.0 | 76.5 | 77.9 | 56.2 | 96.0 |


**W2.2: closed-source models may have restrictions and perform poorly.**

**WR2.2:** To quantify this effect, we measure both evaluation-time refusal/invalid rates and annotation-time refusal rates.

First, we evaluate the behavior of closed-source models during benchmark evaluation. Across 8,000 model responses, explicit refusal occurs in only 70 cases (0.875%), and invalid responses occur in 19 cases (0.2375%). Specifically, GPT-5.4 and Gemini-3-Flash-Preview produce no explicit refusals, while GPT-5.1 produces no refusals with only 0.10% invalid responses. Claude Sonnet 4.6 has a higher refusal rate of 3.50%, but the overall failure rate remains low.

| Model                  | Total responses | Explicit refusals | Refusal rate | Invalid responses | Invalid rate |
| ---------------------- | --------------: | ----------------: | -----------: | ----------------: | -----------: |
| GPT-5.4                |           2,000 |                 0 |        0.00% |                 0 |        0.00% |
| GPT-5.1                |           2,000 |                 0 |        0.00% |                 2 |        0.10% |
| Claude Sonnet 4.6      |           2,000 |                70 |        3.50% |                16 |        0.80% |
| Gemini-3-Flash-Preview |           2,000 |                 0 |        0.00% |                 1 |        0.05% |
| **Overall**            |       **8,000** |            **70** |   **0.875%** |            **19** |  **0.2375%** |

Second, we also analyze refusal behavior during the attribute annotation stage. As shown in the table below, the evaluated closed-source VLMs exhibit very low refusal rates on our sensitive visual content, indicating that they are capable of processing such content for benchmark construction. These results suggest that model-side safety restrictions do not substantially limit the usability of closed-source VLMs in our annotation pipeline.

| Model                  | Refused sections | Section-level refusal rate | Images with ≥1 refused section | Image-level refusal rate |
| ---------------------- | ---------------: | -------------------------: | -----------------------------: | -----------------------: |
| GPT-5.1                |     171 / 99,000 |                     0.173% |                   112 / 11,000 |                   1.018% |
| Gemini-3-Flash-Preview |       4 / 98,406 |                    0.0041% |                     4 / 10934 |                  0.0366% |


Together, these results show that closed-source models are generally capable of processing the sensitive visual content in our benchmark setting, and that refusal or invalid-response behavior is unlikely to be a major factor affecting either benchmark construction or evaluation validity. We will include this analysis to clarify the impact of model-side safety restrictions.


**Q1: In the data-recipe stage, how are the three VLMs selected?**

**A1:** We selected the three VLM annotators based on two considerations: model family diversity and annotation capability. Specifically, we selected representative state-of-the-art models (at that time) from three different VLM families: GPT-5.1, Gemini-3-Flash-Preview, and Qwen2.5-VL-72B-Instruct. For each family, we chose the strongest available model to maximize visual understanding capability and reduce the risk that the annotation quality is limited by a weaker model.

After the initial selection, we further performed manual validation on their attribute extraction capability using representative samples covering different safety categories. Models that demonstrated reliable attribute-level annotation performance were retained for the data construction pipeline. The three models were then combined using an equal-weight field-level majority voting strategy, without performance-based weighting, to avoid introducing additional model-specific bias.

We acknowledge that diversity across model families does not guarantee fully independent errors. Therefore, we additionally conduct attribute-level human validation to verify the reliability of the generated attributes and will clarify the model selection criteria and validation procedure in the revised manuscript.


****

**Q2: The sexual pictures may be illegal.**

**A2:** We clarify that while PolicyShiftBench contains some NSFW-related visual content for safety evaluation purposes, these samples are collected from publicly available datasets and existing benchmark resources and are not obtained through private scraping, unauthorized collection, or distribution of illegal content.

The purpose of including such content is solely to evaluate whether a model can make policy-aware safety decisions under different safety specifications. We do not create, modify, or distribute any sensitive content beyond the original sources. The image sources used in PolicyShiftBench are summarized below:

| Source dataset               | Usage in PolicyShiftBench                                      |
| ---------------------------- | -------------------------------------------------------------- |
| UnsafeBench                  | Unsafe visual safety examples                                  |
| LLaVA-Guard                  | Vision-language safety evaluation samples                      |
| VisionHarm-500K              | Harmful content categories including sexual and unsafe content |
| NeuralShell Gore-Blood       | Blood/gore-related safety examples                             |
| AdImageNet-derived subset    | Category-specific safety examples                              |
| DiffusionDB                  | Safe-control image examples                                    |
| ShareGPT-4o-Image            | Safe-control image examples                                    |
| BLIP3o Pretrain Long Caption | Safe-control image examples                                    |

All data sources are publicly available benchmark or research datasets, and we follow their original licenses and usage policies. In particular, we do not collect user-generated private images, personal sexual content, or non-consensually obtained materials. The sensitive images are only used as benchmark inputs for evaluating model safety behavior and are not intended for redistribution as standalone content.

We will further clarify the data provenance, usage purpose, and ethical considerations of sensitive visual samples in the revised manuscript.

****

**Q3: How about combining Specialized Guardrails with Closed-source MLLMs?**

**A3:** 
To investigate whether combining a specialized guardrail model with a closed-source MLLM can achieve comparable performance, we evaluate hybrid systems built from GuardReasoner-VL-7B and Gemini-3-Flash-Preview. Specifically, we consider two decision aggregation strategies: OR and AND, where the final prediction is obtained by combining the decisions from the two models. We also evaluate a GuardReasoner-gated execution strategy. Note that when both model outputs are valid, gated execution produces the same decision as AND; therefore, it serves as a routing and cost optimization strategy rather than an independent accuracy method.

| Rule / execution                       |      Acc |       F1 |      PSS | Closed routing | End-to-end latency (ms) |
| -------------------------------------- | -------: | -------: | -------: | -------------: | ----------------------: |
| Specialized OR Closed                  |     59.6 |     68.2 |     22.2 |           100% |    9,263.8 (sequential) |
| Specialized AND Closed                 |     66.3 |     55.5 |     34.5 |           100% |    9,263.8 (sequential) |
| AND-equivalent gated execution         |     66.3 |     55.5 |     34.5 |          65.9% |                 7,232.0 |
| **PolicyShiftGuard-7B (single model)** | **78.1** | **76.9** | **72.1** |         **0%** |               **163.9** |

The results show that simple combinations of specialized guardrails and closed-source MLLMs do not recover the policy-shift sensitivity targeted by PolicyShiftGuard. Compared with PolicyShiftGuard-7B, the OR and AND combinations achieve substantially lower PSS, with gaps of 49.9 and 37.6 points, respectively. Although gated execution reduces the number of closed-model calls and lowers deployment cost, it does not improve the underlying decision capability, as it remains equivalent to the AND strategy when both outputs are available.

Furthermore, the latency gap remains substantial. Even with parallel execution, the two-model hybrid system requires approximately 5.92–6.01 seconds per split, compared with only 163.9 ms for PolicyShiftGuard-7B. These results suggest that simply cascading or ensembling a specialized guardrail with a closed-source MLLM cannot replace a policy-conditioned model specifically trained to adapt its decisions under changing policies.

---

Overall, we appreciate the reviewer’s constructive suggestions on benchmark reliability, scalability, and practical evaluation. We will revise the manuscript to further clarify the data construction process, strengthen the validation of benchmark quality, and discuss the scope and applicability of PolicyShiftBench more explicitly.


## AC


We sincerely thank the AC for summarizing the key concerns. We have addressed these concerns through additional analyses and experiments, focusing on four aspects: (1) the scope and validity of policy generalization, (2) the alignment between BP-Adapt and PSS, (3) benchmark scale and reliability, and (4) practical evaluation and reproducibility.

**First, regarding policy generalization**, beyond the original Shift split, we provide additional evidence through evaluations on independently constructed benchmarks and policy-format variations (identifier removal and policy rephrasing), demonstrating robustness beyond specific policies and templates.

**Second, regarding the BP-Adapt and PSS alignment**, we clarified that this alignment is an intentional design choice that directly targets the core capability of PolicyShiftGuard: adapting decisions for the same image under changing policies while satisfying latency constraints. We further validated that this capability is not limited to the original evaluation setting through additional generalization experiments.

**Third, regarding benchmark validity and scale**, we conducted attribute-level human validation of VLM-derived annotations, analyzed evaluation-scale concerns through image-disjoint evaluation and PolicyShiftBench-PLUS, and clarified that the benchmark design prioritizes policy-conditioned diversity rather than only image diversity. We also analyzed the Privacy/PII failure case and identified it as a challenging OOD policy-shift scenario caused by a highly restrictive and counter-intuitive policy formulation.

**Finally, regarding practical evaluation**, we provided detailed closed-source model evaluation protocols, measured refusal and invalid-response rates, and compared hybrid specialized-guardrail/closed-source MLLM systems with PolicyShiftGuard to further validate its efficiency and effectiveness.

Overall, we appreciate the AC’s guidance, which helped us clarify the scope of our claims, strengthen the empirical validation, and better articulate the limitations and applicability of PolicyShiftBench and PolicyShiftGuard.