# AdaReasoner / GenReasoner Rebuttal 当前工作总结

## 1. 论文修改内容列表

完整修改列表：

```text
/data/workspace/code/AdaReasoner/.agent/ref/reviews/PAPER_REVISION_CHANGELOG.md
```

### 正文修改

#### Abstract

- 明确引用并区分 ICLR 2026 AdaReasoner 会议版。
- 明确期刊新增内容：
  - Adaptive Learning；
  - randomized TC/TG study；
  - V*/HRBench 扩展评测。
- 将 “new tasks/new tools” 收窄为：
  - interface robustness；
  - cross-stage transfer；
  - benchmark-level transfer。
- 将“全面超过 GPT-5”收窄为结构化 VSP/Jigsaw 场景。
- 明确工具引入额外 test-time cost 和 failure surface。

#### Introduction

- 不再把 trajectory curation、Tool-GRPO 和工具套件重新声称为期刊新贡献。
- 明确会议版继承内容和期刊增量。
- 三条贡献重新聚焦于：
  1. interface-randomized Adaptive Learning；
  2. randomized training study；
  3. broader evaluation 与可靠性分析。
- 删除 unrestricted zero-shot generalization 表述。

#### Related Work

- 明确 CogCoM、TACO、GRPO、randomization/paraphrasing 的已有先例。
- 将 Tool-GRPO 定位为 multi-turn GRPO instantiation，而非新优化算法。
- 加入 DeepEyes/PixelReasoner 未针对当前 multi-tool interface 适配的公平性说明。

#### Method

- 标明 trajectory curation、TC、Tool-GRPO、reward 和 tool server 来自会议版。
- 将 Adaptive Learning 明确为期刊方法增量。
- 明确 randomization 改变接口而不改变功能。
- 删除 asymmetric reward 已证明 cost-aware abstention 的说法。
- 明确当前 reward 只被证实是有效的 bounded reward shaping。

#### Experiments

- 单任务 TC/TG 实验明确标为 inherited conference context。
- 澄清：
  - “new tasks”实验实际是 cross-stage transfer；
  - “new tools”实验主要是 interface robustness。
- AStar 实验明确区分：
  - TC 未见；
  - inference-only 条件；
  - RL exposure 条件。
- 明确主表 closed models 是 no-tool/single-turn，`+Tools` 才是多轮工具协议。
- 对 DeepEyes/PixelReasoner 增加未适配接口的限制说明。
- 区分 syntactic tool success 与 semantic correctness。
- 将 repeatability、latency、failure、judge audit 指向附录。

#### Discussion and Limitations

新增专门章节，覆盖：

- interface-level 而非 unrestricted generalization；
- latency 和 test-time compute；
- Point/Molmo 等外部专家模型依赖；
- tool observation 错误传播；
- 手工 task blueprint；
- structured 与 open-ended task 的差异；
- judge 和 baseline protocol 的限制；
- fixed-checkpoint inference repeat 不等于 training-seed variance。

#### Conclusion

- 重写为期刊增量逻辑。
- 删除“工具普遍取代模型规模”的绝对说法。
- 明确：
  - structured tasks 上工具增益很大；
  - open-ended tasks 仍依赖 base model；
  - 工具重新分配而不是消除错误。

### 附录修改

- 增加 Jigsaw source-image 先划分、后构造 patch 的明确说明。
- 增加 closed/no-tool 与 `+Tools` protocol 说明。
- 修正 Qwen2.5-72B-Instruct judge 说明。
- 增加 fixed-checkpoint stochastic repeats 表。
- 增加 latency stage decomposition 表。
- 增加 AStar/Point per-call latency。
- 增加 early-turn fault injection 表。
- 增加 500-item judge semantic audit。
- 增加 response-length quartile 描述性检查。
- 说明 `332,649` 是 `max_samples` 配置上限，不是经核验的唯一 trajectory 数。

---

## 2. 论文修改和编译状态

唯一论文工作目录：

```text
/data/workspace/code/AdaReasoner/.agent/ref/69a66abdb76ba160fb253194
```

所有可见新增和改写使用：

```latex
\red{...}
```

或：

```latex
\begin{revision}
...
\end{revision}
```

论文已成功编译：

```text
/data/workspace/code/AdaReasoner/.agent/ref/69a66abdb76ba160fb253194/main.pdf
```

最新 PDF SHA256：

```text
93d95f7708a7b1722d9889622c41732024b2c5b9b828f6c25505e44fe89a30f8
```

没有 undefined citation/reference 错误，只剩少量原有的 float/underfull 排版 warning。

论文仓库本地提交：

```text
c0dd6d3 revise manuscript for journal rebuttal
a8ff947 clarify judge audit scope
```

---

## 3. `.agent/ref` 整理结果

已删除过时和相互冲突的材料：

- 旧版 `exps_we_have.md`
- 旧 plans 目录
- 旧 split reviewer responses
- 旧 general response
- 旧 dashboard
- 无关的 PolicyShiftGuard rebuttal reference
- 重复的 local original drafts

当前 reviews 目录只保留：

```text
CITATION_iclr2026.md
E5_tool_failure_results.md
ICLR VERSION.pdf
PAPER_REVISION_CHANGELOG.md
RESULTS_TABLE.md
reviews.md
requirments.md
rebuttal_content/POINT_BY_POINT_REBUTTAL.md
rebuttal_content/judge_audit_500_selected_20260804.tar.gz
```

新的唯一证据状态表：

```text
/data/workspace/code/AdaReasoner/.agent/ref/reviews/RESULTS_TABLE.md
```

状态表已经更新为当前事实：

- WebMMU 三次完整产物已经存在并核验。
- HRBench 仍未冻结。
- 200-item VSP failure summary 因缺 raw artifact，暂不进入论文。
- 当前论文使用可本地审计的 100-item VSP + 100-item Jigsaw fault experiment。
- 不再把 judge audit 写成人类双盲实验。
- 不再把 `332,649` 当成 unique sample count。

---

## 4. Rebuttal 重写状态

文件：

```text
/data/workspace/code/AdaReasoner/.agent/ref/reviews/rebuttal_content/POINT_BY_POINT_REBUTTAL.md
```

已从约 6,700 词压缩到约 2,200 词，并完成：

- 补回缺失的 R1-5。
- 修复原稿 R1-4 答非所问。
- 删除 “We agree” 式低姿态表达。
- 统一 conference/journal boundary。
- 统一 generalization 范围。
- 统一 GPT-5 matched-protocol 数字。
- 统一 fault 表和 baseline。
- 更新 WebMMU 当前完整结果。
- 明确 inference repeat 并非 training seeds。
- 将 judge audit 准确称为 single-reviewer semantic audit。
- 明确 audit 未包含 GenReasoner 答案，因此不能直接证明 tool verbosity 无偏。
- 不再声称 pHash/CLIP 已完成。
- 不再声称 200-item failure experiment 已有完整证据。
- 语气改成事实驱动、直接回应，不卑不亢。

AdaReasoner 主仓已提交并成功推送：

```text
53fad2a consolidate rebuttal state and evidence
912164d tighten judge audit claims
```

GitHub 当前已同步到 `origin/main`。

---

## 5. 仍需完成的工作

### 必须补充

#### 5.1 HRBench 统一重判

当前旧结果：

```text
63.04 ± 0.14
```

每次实验有 108–111 条 `Z` fallback，原因是外部 answer extractor 不可达。

现有明确格式恢复只能证明：

```text
≥68.92 ± 0.14
```

正式提交前需要对全部 fallback 做统一、离线、可复现重判，然后冻结最终数字。

#### 5.2 Matched-budget accuracy–latency curve

目前完成的是：

- stage decomposition；
- per-tool latency；
- wall time。

这些结果能证明 CPS 不是成本 proxy，但还不能完整回答 reviewer 要求的 matched-budget curve。

当前论文已经主动承认这一点，并删除 universal efficiency claim。若时间允许，仍应补充该曲线。

#### 5.3 Jigsaw near-duplicate 检查

代码能证明：

- 先选择不同 COCO source images；
- 然后分为 SFT/RL/test；
- 再构造 patch；
- 因此 exact source image 是 disjoint 的。

但 reviewer 额外要求 pHash+CLIP near-duplicate screening。完整 source images 当前机器上不存在，因此没有声称该检查已经完成。

#### 5.4 Judge audit 的独立人类确认

现有 500 条是 Codex single-reviewer semantic audit，不是两个人类作者盲标。

如果正文要使用 “human validation”，仍需：

- 作者盲审确认；
- 或增加第二名独立人类标注者。

否则维持当前的 single-reviewer semantic audit 表述即可。

#### 5.5 实际 cold-start 数据分解

`332,649` 实际来自 LLaMA-Factory 配置：

```yaml
max_samples: 332649
```

它不是数据集唯一 trajectory 数。

正式 release manifest 需要给出：

- post-filter 实际样本数；
- VSP/Jigsaw/GUIQA 分解；
- reflection/failure/direct 等分解。

### 可选增强

- late-turn fault injection；
- with/without failure/reflection trajectory training ablation；
- symmetric versus asymmetric reward ablation；
- easy/no-tool control；
- 将 GenReasoner 答案加入 judge verbosity audit。

---

## 6. 推送状态

AdaReasoner 主仓已经成功推送。

Overleaf 论文仓库已成功推送：

```text
main...origin/main
```

已推送提交：

```text
c0dd6d3 revise manuscript for journal rebuttal
a8ff947 clarify judge audit scope
```

远端更新范围：

```text
f90f281..a8ff947  main -> main
```

Overleaf token 只通过临时 askpass 文件使用；推送完成后，临时 token 和
askpass 文件已覆写并删除，未写入仓库或记忆文件。

---

## 7. 2026-08-10 最新核验：主表对齐、行文和引用位置

- 所有改动改为局部标红：改哪句标哪句，改哪个数字标哪个数字。
- 最大最终主表 `tex/tables/final_main.tex` 是 Qwen2.5-VL-7B base row 的唯一
  对齐源：
  - VSPO `25.39`
  - VSP `28.09`
  - Jigsaw `45.70`
  - BLINK-J `52.67`
  - GUIChat `68.09`
  - WebMMU Act. `67.48`
  - HRBench `63.62`
  - V* `63.35`
  - Avg. `51.80`
- 单任务表已恢复 GUIChat/WebMMU，不再通过删列回避冲突。
- Qwen2.5-VL-3B 的 GUIChat/WebMMU Act. 对齐为 `46.26/54.47`。
- 7B 详细 WebMMU Avg. 为 `58.36`，对应
  Act./Comp./Reason. `67.48/69.31/48.46`。
- 论文中已删除“因为不同协议所以删列/结果不一致”“conference artifact”
  等主动暴露问题的解释。
- AdaReasoner 会议版引用和 conference/journal boundary 只保留在
  Related Work；Abstract、Introduction、Method、Experiments、caption 和
  Conclusion 均不再引用会议版。
- rebuttal 的 R2-3 已改为直接说明：所有 active base rows 已对齐最终主表，
  不再解释历史协议问题。
- 论文提交 `7bb89dc` 已推送到 Overleaf：
  `ea9bd7f..7bb89dc  main -> main`。
- 最终本地编译成功，共 29 页；PDF SHA256：
  `2acaf57986e92ff442f6fbde47aabb99b2729cff14625bb7f31c5a3409681d77`。

## 8. 2026-08-10 主版本回退与 alter 归档

- `7bb89dc` 对应的整套修改版已另存为可独立编译的
  `main_alter.tex`，并为它复制了全部受影响的 `_alter.tex` section/table。
- 正式 `main.tex` 及其原始 section/table 已逐字节恢复到 `ea9bd7f`
  （即 `7bb89dc` 修改之前的状态）。
- 两个入口都已编译通过：
  - `main.tex`：31 页，SHA256
    `bf1f90caad05b898ca169b6ead05527abe16fddac799b1704bf7b1063d934aae`
  - `main_alter.tex`：29 页，SHA256
    `6efc936859c9043f86ff2998aff79ce77593b9c4acb3158e2ed1cf73c224a9ec`
- Overleaf 已推送提交：
  `4e90810 archive alternate revision and restore main manuscript`。
