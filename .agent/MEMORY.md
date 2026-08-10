# AdaReasoner Working Memory

## Canonical manuscript repository

- The only manuscript working tree is:
  `/data/workspace/code/AdaReasoner/.agent/ref/69a66abdb76ba160fb253194`
- All future manuscript edits, compilation checks, commits, and Overleaf pushes must be performed in that repository.
- After each completed manuscript-editing task:
  1. inspect `git diff`;
  2. compile the paper;
  3. commit the manuscript changes;
  4. push `main` to the Overleaf remote.
- If a push is blocked by authentication or network state, keep the local commit intact and report the exact blocker rather than claiming that the push succeeded.

## Revision-marking rule

- All visible manuscript text added or rewritten for the current revision must be shown in red.
- Mark only the sentence, phrase, or number that actually changed; do not color
  an untouched surrounding paragraph or an entire section.
- Use `\red{...}` for local replacements. Avoid broad `revision` environments
  unless every enclosed sentence is genuinely new.
- Purely technical, non-visible build changes do not need red marking.

## Evidence-placement and tone rule

- The AdaReasoner conference citation and the conference/journal boundary
  belong only in Related Work, not in the Abstract, Introduction, Method,
  Experiments, captions, or Conclusion.
- Do not proactively expose weaknesses, protocol mistakes, or defensive
  explanations outside the dedicated Discussion and Limitations section.
- Remove wording such as “because protocols differed,” “not comparable,”
  “conference-era artifact,” or “we omitted/excluded because...”.
- Keep the argument direct, factual, and firm. Do not write “we agree” or
  flatter reviewers.
- Detailed reliability, latency, fault-injection, and judge-audit evidence
  should go to the appendix unless essential to the main narrative.
- No result may be described as completed unless its configuration and artifact
  are locally auditable.

## Canonical rebuttal source

- The active point-by-point response is:
  `.agent/ref/reviews/rebuttal_content/POINT_BY_POINT_REBUTTAL.md`
- Older split reviewer drafts are not authoritative once the point-by-point response has been updated.

## 2026-08-10 manuscript revision

- Red-marked journal-rebuttal manuscript revision committed in the canonical
  Overleaf working tree as commit `c0dd6d3` (`revise manuscript for journal rebuttal`).
- Local compilation succeeded; generated PDF SHA256:
  `3869d3e0641700ef377a509f5e67012c6f2bbd0f44d5921953e0f6f4125442a9`.
- `git push origin main` was attempted immediately after the commit but failed
  because the current environment has no Overleaf password/token available:
  `fatal: could not read Password for 'https://git@git.overleaf.com'`.
- The local manuscript branch is therefore one commit ahead of `origin/main`.
  On the next manuscript task, retry the push before making additional edits.
- Follow-up manuscript audit clarification committed as `a8ff947`
  (`clarify judge audit scope`). The Overleaf branch is now two commits ahead
  of `origin/main`; push retry again failed for the same missing-password
  condition. Latest compiled PDF SHA256:
  `93d95f7708a7b1722d9889622c41732024b2c5b9b828f6c25505e44fe89a30f8`.

## 2026-08-10 Overleaf push completed

- Commits `c0dd6d3` and `a8ff947` were successfully pushed to the canonical
  Overleaf repository.
- The manuscript working tree is synchronized:
  `main...origin/main`.
- The supplied Overleaf token was used only through a temporary askpass file;
  the temporary token and askpass files were overwritten and removed after
  the push. Do not store the token in the repository or memory files.

## 2026-08-10 current cross-table and narrative rule

- The largest final main table (`tex/tables/final_main.tex`) is the canonical
  source for the Qwen2.5-VL-7B base row:
  - VSPO 25.39
  - VSP 28.09
  - Jigsaw 45.70
  - BLINK-J 52.67
  - GUIChat 68.09
  - WebMMU Act. 67.48
  - HRBench 63.62
  - V* 63.35
  - Avg. 51.80
- Every active overlapping Qwen2.5-VL-7B value must match this row and each
  corrected number must be individually red.
- The single-task table retains GUIChat and WebMMU; do not hide the columns or
  explain past discrepancies in the paper.
- Qwen2.5-VL-3B GUIChat/WebMMU Act. are aligned to 46.26/54.47.
- The detailed WebMMU Avg. for the 7B base row is 58.36, computed from
  Act./Comp./Reason. 67.48/69.31/48.46.
- Manuscript commit `7bb89dc` (`align base tables and tighten revision
  narrative`) implements these rules and was pushed to the canonical Overleaf
  remote. The temporary Overleaf askpass credential file was securely removed.
- Final local compilation succeeded (29 pages); PDF SHA256:
  `2acaf57986e92ff442f6fbde47aabb99b2729cff14625bb7f31c5a3409681d77`.

## 2026-08-10 alternate manuscript archive and main restoration

- The revision introduced in manuscript commit `7bb89dc` was judged
  unsatisfactory and is no longer the canonical `main.tex`.
- That complete version is preserved as an independently compilable alternate:
  `main_alter.tex`, with `_alter.tex` copies of every changed section/table it
  depends on.
- `main.tex` and every original section/table changed by `7bb89dc` were
  restored byte-for-byte to commit `ea9bd7f` (the immediate pre-`7bb89dc`
  state).
- Both entry points compile:
  - restored `main.tex`: 31 pages, PDF SHA256
    `bf1f90caad05b898ca169b6ead05527abe16fddac799b1704bf7b1063d934aae`;
  - archived `main_alter.tex`: 29 pages, PDF SHA256
    `6efc936859c9043f86ff2998aff79ce77593b9c4acb3158e2ed1cf73c224a9ec`.
- Manuscript commit `4e90810` (`archive alternate revision and restore main
  manuscript`) was pushed to Overleaf.
