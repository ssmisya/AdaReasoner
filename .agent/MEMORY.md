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
- Use `\red{...}` for short replacements and the `revision` environment for multi-paragraph text, lists, and tables.
- Purely technical, non-visible build changes do not need red marking.

## Evidence-placement rule

- Main-text changes should preserve the paper's argument:
  conference/journal boundary -> scoped journal contribution -> method -> core experiments -> bounded conclusions.
- Detailed reliability, latency, fault-injection, judge-audit, and protocol evidence should go to the appendix unless it is essential to the main narrative.
- No result may be described as completed unless its configuration and artifact are locally auditable.

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
