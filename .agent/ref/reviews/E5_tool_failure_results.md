# E5 Candidate Follow-up: Controlled Early-Turn Tool-Failure Injection (VSP)

> **Status:** The numerical summary was added on 2026-08-10, but the
> corresponding `E5_*_v2` raw directories are not present in this checkout.
> It is therefore **not** the fault table used in the manuscript or canonical
> rebuttal. The auditable 100-item VSP-navigation and 100-item Jigsaw results
> remain the current evidence in `RESULTS_TABLE.md`.

**Model:** AdaReasoner-7B (with full tool set incl. Point) · **Subset:** 200 items (navigation 100 + verify 100) · **Injection:** early turn · **Clean baseline: 0.89** (full-suite VSP result).

## Results

| Fault | Overall VSP accuracy | Δ from clean 0.89 | Navigation | Verify |
|---|---:|---:|---:|---:|
| **clean (no fault)** | **0.89** | — | **0.87** | **0.98** |
| plausible-but-wrong | 0.45 | −0.44 | 0.42 | 0.48 |
| missing | 0.465 | −0.425 | 0.27 | 0.66 |
| malformed | 0.42 | −0.47 | 0.18 | 0.66 |
| timeout | 0.39 | **−0.50** | 0.22 | 0.56 |
| contradictory | 0.42 | −0.47 | 0.29 | 0.55 |

## Takeaways

- Under a genuinely tool-dependent setup (clean 0.89), every injected runtime tool failure causes a large accuracy drop of **−0.42 to −0.50**, rather than being negligible.
- The **navigation** subtask is by far the most fragile: it collapses from 0.87 (clean) to 0.18–0.42 under fault, because navigation relies on the tool's path/coordinate output. **verify** degrades less (0.98 → 0.48–0.66) since it depends less on tool correctness.
- **timeout** (−0.50) and **malformed** (−0.47) are the most damaging (tool effectively unusable); **plausible-but-wrong** (−0.44) is comparatively less severe, as the model occasionally detects the inconsistency.
- These results directly evaluate runtime failure (plausible-wrong / missing / malformed / timeout / contradictory, injected early) rather than inferring it from an irrelevant-tool distractor.

## Notes

- Each fault condition injected on all 200 evaluated items (early-turn).
- Model: `AdaReasoner-7B-Randomized`, tools: `AStarWithPixelCoordinate, Draw2DPath, Point`, `max_rounds=6`.
- Raw logs/results: `rebuttal_exps/E5_{baseline,plausible_wrong,missing,malformed,timeout,contradictory}_v2/`.
