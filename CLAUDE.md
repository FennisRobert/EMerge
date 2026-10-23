# CLAUDE.md: working on the `hertzon` branch of EMerge

This branch is the version of EMerge used by the Hertzon GUI. EMerge itself is maintained by
Robert Fennis on `main`. EMerge is also a learning project for Robert: he wants to write the
core FEM and EM code himself. These rules keep `hertzon` close to `main`, so that fixes keep
flowing in both directions without blocking Robert's own work.

## Where a change goes

| Kind of change | Where it goes |
|---|---|
| Bug fix in EMerge itself (wrong results, crashes, broken API) | **Always** a PR to `main` |
| Change that only matters for the Hertzon GUI | Only on `hertzon`, never a PR to `main` |
| Improvement to code tagged as AI written (see below) | Free to PR to `main` |
| New technical feature in EMerge's core (FEM, EM theory, solvers, elements, assembly) | Keep it on `hertzon`. A PR to `main` is allowed but discouraged, and may be rejected. |

### 1. Bug fixes in EMerge: always open a PR to `main`
- Any fix to EMerge's own behaviour must go to `main` as a PR, so the two branches don't drift apart.
- Keep each PR to one fix. Describe the bug, how to reproduce it, and the fix.

### 2. GUI-specific changes: keep them on `hertzon`
- If a change exists only because the GUI needs EMerge to behave differently, and it doesn't improve EMerge for its other users, keep it on `hertzon`.
- Keep it as separate as possible (see "Add, don't rewrite").

### 3. AI-written changes to EMerge must be marked
- You may use your own skills to change EMerge code if, and only if, the change improves EMerge.
- Every function, class or module you write or substantially rewrite must carry the tag that is already used in this repository:
  - Python: `# [Written by Claude]` on the line above the `def` or `class`.
  - Rust: `// [Written by Claude]` on the line above the item.
  - A whole new module: one line at the top, `# [Written by Claude] All code in this module was written by Claude (AI).` (or `//` in Rust).
- Never remove an existing tag. Never add a tag to code you did not write.

### 4. New technical features: elaborate PR, or keep them on `hertzon`
- If you add new technical capability to EMerge (for example new element types, isoparametric geometry, new solvers, new assembly or EM formulations), keep it on `hertzon` by default.
- Robert wants to implement this kind of work himself, so such PRs are **not encouraged** and may be rejected even if they are correct.
- If you still open one, write a clearly elaborated PR: what it adds, the theory behind it, how it was validated, and which files it touches.

### 5. Code tagged as AI written is free to improve
- Code already tagged `[Written by Claude]` (for example the AURA mesh refinement in `src/emerge/_emerge/aura/` and `emr/src/aura/`, and the adaptive frequency sweep in `adaptive_sweep.py`) is not part of Robert's learning work.
- Improvements to tagged code may be sent to `main` as PRs without restriction.
- To find tagged code, run `grep -rn "\[Written by Claude\]" src emr`.

## Add, don't rewrite

Whenever a change on `hertzon` touches EMerge's core back end, add **new** code next to the existing code instead of editing it in place. Every bug fix that lands on `main` then merges into `hertzon` without conflicts.

- **Assembler, kernels and elements:** if the assembly, a Rust kernel or a basis function must work differently, write a new function or file next to the old one. For example, add `curlcurl_iso.rs` next to `curlcurl.rs`, or a new assembler method. Then route the call to it through a setting or a small dispatch at the call site. Leave the original function unchanged.
- **Rust:** add new `.rs` files and register them, instead of editing the existing kernels in `emr/src`.
- **Public API:** existing user scripts must keep working. Add options or new functions; don't change existing signatures or defaults.
- **Edits to existing core files:** keep them as small as possible, ideally a few lines that switch to your new code.

## Git workflow

- **Work for `hertzon`:** commit and push directly to `hertzon`.
- **A fix or PR for `main`:**
  - Start a new branch from `main`, **never from `hertzon`**: `git fetch origin && git switch -c fix/<short-name> origin/main`.
  - A branch made from `hertzon` would carry all of the `hertzon` work into the PR.
  - If the fix already exists on `hertzon`, `git cherry-pick` it onto the new fix branch.
- **After a PR is merged:** bring `main` into `hertzon` with `git switch hertzon && git merge origin/main`. Merge `main` into `hertzon` regularly in any case, so conflicts stay small.
- **Never push to `main` or `v3.0-dev`.** Both are protected, and changes reach them only through approved PRs.
- **Never change** the version in `pyproject.toml` or `.bumpversion.toml`, the release workflow `.github/workflows/release.yml`, or tags. Releases are Robert's.

## Building and testing

- The Rust kernels (`emr/`) are built into the package with maturin. After changing Rust code, rebuild with `uv sync --inexact`, not `maturin develop`.
- Run the tests with `uv run pytest`. A PR to `main` must not break existing tests.
- Add a test for every bug fix.
