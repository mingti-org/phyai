---
name: phyai-solve-pr-comments
description: Triage and resolve review comments on a GitHub PR — fetch all comment surfaces (issue / inline / review), validate each suggestion against upstream source rather than trusting blindly, present findings to the user before editing, then make focused changes and re-run tests. Use when the user asks to "check PR N's comments", "address review feedback", "resolve PR comments", or similar.
---

# Solve PR Comments

The point of this skill is to **resolve PR review feedback correctly, not just compliantly**. Bot suggestions (gemini-code-assist, copilot, codex) are often partially right or confidently wrong — they pattern-match local code without checking the upstream library contract. Your job is to verify each claim against ground truth, then surface the verdict to the user before touching code.

## When to use

User says one of:
- "check PR N's comments"
- "address review feedback on PR N"
- "resolve / fix PR N comments"
- "look at the comments on PR N"
- "there are comments on PR N, fix them"

Skip this skill for:
- Asking what a comment **says** (just `gh pr view`)
- Re-running existing review (use `/review` instead)

## Workflow

### 1. Fetch every comment surface — don't trust `gh pr view --comments`

`gh pr view N --comments` quietly drops content when GitHub's GraphQL flags Projects-classic deprecation. Hit the REST endpoints directly:

```bash
# Conversation comments (issue thread)
gh api repos/OWNER/REPO/issues/N/comments \
  --jq '.[] | {id, user: .user.login, body, created_at}'

# Inline review comments (the line-anchored ones)
gh api repos/OWNER/REPO/pulls/N/comments \
  --jq '.[] | {id, user: .user.login, path, line, side, body, created_at}'

# Overall reviews (the summary block on top of each review)
gh api repos/OWNER/REPO/pulls/N/reviews \
  --jq '.[] | {id, user: .user.login, state, body, submitted_at}'
```

If `gh auth status` says "token invalid" — stop and ask the user to re-auth. Don't try to work around it.

### 2. Triage before doing anything

For each comment, classify:

| Class | Examples | Action |
|---|---|---|
| **Real bug / perf issue** | "race condition in shared dict", "per-forward fp32 cast" | Plan a fix, verify approach with user |
| **Wrong claim** | Bot extrapolated from a pattern that doesn't apply to this code | Reject — explain to user *why* it's wrong (cite upstream source) |
| **Doc-only** | Subtle contract that's not visible from code | Add docstring/comment, no code change |
| **Style / nit** | Naming, formatting | Skip unless user asks; not the point of review |
| **Already fixed** | Comment is on stale code | Mark as outdated, move on |

### 3. Validate every claim against ground truth

This is the highest-leverage step. Before believing a bot:

- **Library contract claims** ("X needs fp32") -> read the upstream source. CUDA kernels usually live in `.../site-packages/<pkg>/data/csrc/*.cu`. Look for `TORCH_CHECK` / `TVM_FFI_ICHECK` / dispatch macros.
- **Reference implementations** -> grep `.tmp/<reference-project>/` (e.g. sglang, vllm, lerobot) for how mature codebases handle the same kernel. SGLang's `python/sglang/srt/layers/` is a particularly good reference for kernel wrappers. **If the repo you need isn't already under `.tmp/`, clone it** — `git clone --depth 1 <upstream-url> .tmp/<name>` is enough for a read-only consult; full history is rarely needed. Don't try to reason from training-data memory of the source.

  **STRICT — DO NOT COPY CODE FROM REFERENCE REPOS.** They exist *only* for correctness verification: confirming a kernel's dtype contract, double-checking a math formula, comparing dispatch logic. The fix you write must be authored from scratch in this project's style and abstractions. Pasting an SGLang/vLLM/lerobot block — even one that "looks like it fits" — is a license violation, breaks our layer hierarchy, and drags in dependencies we don't want. Read their code, understand the constraint, then write our own.
- **Project tests** -> read existing tests under `tests/` before editing. They encode the contract the user already expects.

A bot's claim is worth nothing until you've checked it against the upstream contract. The bot may have pattern-matched a similar-looking issue from a different library; the right fix here may be the opposite of what they suggested, or there may be no fix at all (and the contract just needs to be documented).

### 4. Present findings *before* editing

Reply to the user with a tight table:

```
| # | File:Line | Issue | Verdict | Plan |
|---|---|---|---|---|
| 1 | foo.py:100 | claim X | Real | Pre-allocate fp32 in __init__ |
| 2 | bar.py:50  | claim Y | Wrong | Bot misread; weight dtype must match input. Add docstring note instead. |
| 3 | baz.py:200 | claim Z | Stale | Already fixed in commit abc123 |
```

Wait for the user to confirm scope. Do not silently apply every bot suggestion.

### 5. Make the changes — minimal and focused

- One concept per edit. Don't bundle "address comment + refactor + rename".
- Update docstrings for any *contract* you discovered (e.g., "flashinfer requires weight dtype == input dtype"). Future readers shouldn't have to re-derive what you just learned.
- If a bot suggestion has a code block (a `suggestion:` block in the comment body), still *read* it but adapt to local conventions; don't paste verbatim if it conflicts with project style.

### 6. Verify before reporting done

- Run the relevant test file (`pytest tests/.../test_X.py`) — not the whole suite, just the affected module.
- For perf changes (no parity test): write a tiny smoke test that exercises both old/new paths and asserts numerical match against a `torch.nn.functional` reference.
- For dtype changes: think about the load path too. `Tensor.copy_` silently casts — does that propagate precision loss anywhere? Add a placement-load warning if surprising.

### 7. Reply on the PR (only if user asks)

If the user wants to push changes back as PR review replies:
```bash
gh api repos/OWNER/REPO/pulls/comments/COMMENT_ID/replies \
  -f body="Addressed in commit SHA — explained reasoning in the diff."
```
Don't auto-reply. The user may want to sanity-check before publishing.

## Anti-patterns

- **Pasting a bot's `suggestion:` block verbatim** without checking whether the project already has a different pattern for that situation.
- **"All five comments are valid, fixing all"** — at least one is usually wrong or stale. If you don't find a wrong one, you didn't read carefully.
- **Treating "medium priority" as "must-fix"** — the bot's priority labels are heuristic. A medium-priority race condition is more important than a high-priority style nit.
- **Editing without reading the test file first** — tests document the *intended* behavior; an edit that breaks the test is usually wrong even if it matches the bot's suggestion.
- **Silent dtype/shape changes** — if your fix changes the dtype or shape of a parameter, the load path (`apply_placements` / `state_dict` loader) needs to handle the cast and ideally warn on mismatch.

## Quick reference: gh CLI cookbook

```bash
# List PRs needing attention
gh pr list -R OWNER/REPO --search "review:required"

# Full PR snapshot (metadata + body, no comments)
gh pr view N -R OWNER/REPO --json number,title,state,body,author,headRefName

# Diff for the PR
gh pr diff N -R OWNER/REPO

# Get the commit on the PR head
gh pr view N -R OWNER/REPO --json headRefOid -q .headRefOid

# Resolve / mark conversation as resolved (REST not GraphQL — needs separate call)
# (gh CLI doesn't support this directly; use the GraphQL endpoint or the web UI)
```
