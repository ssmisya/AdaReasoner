# GenReasoner Revision Evidence — Current Snapshot (2026-08-10)

This file is the current evidence index for the journal revision. Older draft
plans and split reviewer responses have been removed. The point-by-point reply
is maintained at `rebuttal_content/POINT_BY_POINT_REBUTTAL.md`.

## 1. Results used in the revised manuscript

All repeatability numbers below are three stochastic inference repeats of one
fixed `AdaReasoner-7B-Randomized` checkpoint. They are not training-seed
variance or significance tests.

| Benchmark | Run 1 | Run 2 | Run 3 | Mean ± sample std | Evidence |
|---|---:|---:|---:|---:|---|
| VSP | 89.91 | 89.64 | 88.27 | **89.27 ± 0.88** | `E3_vsp_fixed`, `E2_vsp_fixed_seed2/3` |
| VSPO | 78.98 | 78.32 | 78.62 | **78.64 ± 0.33** | `E3_vspo_full`, `E2_vspo_seed2/3` |
| Jigsaw-COCO | 88.20 | 88.20 | 88.40 | **88.27 ± 0.12** | `E3_jigsaw`, `E2_jigsaw_seed2/3` |
| BLINK-J | 88.00 | 88.67 | 88.00 | **88.22 ± 0.39** | `E_blinkj*` |
| V* | 68.59 | 68.06 | 67.54 | **68.06 ± 0.53** | `E_vstar*` |
| GUIChat, Qwen2.5-72B-Instruct judge | 73.70 | 73.49 | 73.60 | **73.60 ± 0.11** | `E_guichat*/result_judged.jsonl` |
| WebMMU Functional/Act., Qwen2.5-72B-Instruct judge | 72.15 | 71.14 | 71.95 | **71.75 ± 0.53** | `E_webmmu_fix*/result_judged.jsonl` (1,476 samples/run; Functional 492) |

### Not frozen

- **HRBench:** legacy 63.04 contains 108–111 `Z` fallbacks/run from an
  unreachable external answer extractor. Deterministic explicit-answer
  recovery gives only a lower bound of **≥68.92 ± 0.14**. It is excluded from
  the new repeatability table until every fallback is uniformly re-scored.
- The newer `qwen25vl_eval/` matrix contains incomplete and protocol-mismatched
  runs. It is not used in the manuscript or rebuttal unless a row has a valid
  `DONE.json`, matching protocol fingerprint, and final summary audit.

## 2. Latency evidence

| Benchmark | Generation | Tool execution | Orchestration | Other/I/O | Wall time |
|---|---:|---:|---:|---:|---:|
| BLINK-J | 92.48% | 0.39% | 0.01% | 7.11% | 175.9 s |
| Jigsaw | 90.95% | 0.55% | 0.02% | 8.49% | 1,275.8 s |
| GUIChat | 73.85% | 18.65% | 0.01% | 7.50% | 2,744.2 s |
| V* | 31.16% | 48.11% | 0.05% | 20.69% | 1,411.9 s |
| VSPO | 43.83% | 54.50% | 0.01% | 1.66% | 13,427.2 s |
| VSP | 39.11% | 59.42% | 0.01% | 1.47% | 6,541.6 s |

Microbenchmark: AStar **0.092 ms/call** versus Point/Molmo
**255.333 ms/call** (~2,775×). This proves that CPS is not a cost proxy. It
is not a matched-budget accuracy–latency curve.

## 3. Runtime tool-failure evidence

The manuscript uses the locally auditable early-turn experiments:

| Fault | VSP navigation, clean 0.34 | Δ | Jigsaw, clean 0.90 | Δ |
|---|---:|---:|---:|---:|
| plausible-but-wrong | 0.39 | +0.05 | 0.77 | -0.13 |
| missing | 0.36 | +0.02 | 0.73 | -0.17 |
| malformed | 0.29 | -0.05 | 0.84 | -0.06 |
| timeout | 0.28 | -0.06 | 0.81 | -0.09 |
| contradictory | 0.30 | -0.04 | 0.82 | -0.08 |

Evidence: `E5_*`, `E5j_*`, `E5_matrix.json`, `E5j_matrix.json`.
The automatic detect/react heuristic is not reported because it also fires on
the clean baseline. The untracked 200-item full-tool summary is retained as a
candidate follow-up, not as manuscript evidence, because its raw artifacts are
not present in this checkout.

## 4. Judge audit

- 500 sampled; 498 valid.
- Agreement: **90.76% (452/498)**; Wilson 95% CI **87.90%–93.00%**.
- Cohen’s κ: **0.781**.
- Precision/recall/specificity: **95.59% / 91.29% / 89.44%**.
- FP/FN: **15 / 31**.
- GUIChat: **86.29% (170/197)**; WebMMU: **93.69% (282/301)**.
- Length-quartile agreement: 87.10/91.20/93.55/91.20%.

This is a reproducible **single-reviewer semantic audit** archived in
`rebuttal_content/judge_audit_500_selected_20260804.tar.gz`; it is not a
blinded two-human study.

## 5. Manuscript revision status

The canonical paper tree is:
`../69a66abdb76ba160fb253194`.

A red-marked revision has been compiled locally. The change list is
`PAPER_REVISION_CHANGELOG.md`.

## 6. Remaining submission gates

1. Uniformly re-score all HRBench `Z` fallbacks before freezing that row.
2. Run the requested matched-budget accuracy–latency curve, or keep the paper's
   efficiency claim explicitly limited to stage decomposition/per-tool cost.
3. Run pHash+CLIP near-duplicate screening when the complete Jigsaw source-image
   sets are available; the exact-source split construction is already stated.
4. Add an independent human confirmation if the final paper wants to call the
   judge audit “human validation.”
5. Populate the actual post-filter cold-start trajectory count and task
   decomposition from the release manifest. `332,649` is only the configured
   `max_samples` cap.
