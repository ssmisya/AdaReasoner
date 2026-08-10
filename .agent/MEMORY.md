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

## 2026-08-10 red-marking and cross-table correction

- Revision marking was refined from broad multi-paragraph color environments
  to local `\red{...}` units; modified captions/sentences/paragraphs are marked
  independently rather than coloring an entire section by default.
- Reviewer-2's cross-table conflict is now resolved structurally:
  - inherited single-task Table `new_main` retains only VSPO/VSP/Jigsaw/BLINK-J;
  - its GUIChat/WebMMU columns were removed because they used conference-era
    protocols;
  - the active journal generalization and main tables share the journal-wide
    base values: 3B GUIChat/WebMMU = 46.26/54.47 and 7B = 68.09/67.48;
  - detailed generalization WebMMU Avg. was corrected to 58.36 so its category
    mean matches 67.48/69.31/48.46.
- Latest locally compiled PDF SHA256:
  `d17a935ec9808f08ec755cbfc8f63e5da63e3c97b061f9cca303f9e6adaf1b31`.
