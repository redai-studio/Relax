---
name: git-commit
description: Creates git commits following Conventional Commits with an optional scope and explanatory bodies for non-trivial commits. Use when user wants to commit changes, create commit, save work, or stage and commit. Verifies intended staged changes and pre-commit checks before creating local commits.
---

# Git commit

Creates git commits following Conventional Commits. Read `AGENTS.md` for project constraints.

## Recent project commits

!`git log --oneline -5 2>/dev/null`

## Quick start

```bash
# 0. Inspect current state
git status -sb
git diff --cached --stat
git diff --stat

# 1. Stage only the intended changes
git add <files>
git diff --cached --stat

# 2. Run pre-commit checks (MUST do before committing)
pre-commit run --all-files --show-diff-on-failure
# If it fails or auto-fixes files: re-stage with git add, then re-run until clean

# 3. Re-stage intended files (pre-commit may have modified files), then verify index
git add <files>
git diff --cached --stat
git diff --stat

# 4. Create commit from the prepared message
git commit -F /tmp/commitmsg.txt
```

## Staging discipline

- Treat the index as the source of truth for the commit. Use `git diff --cached --stat` and `git diff --cached --name-status` before writing the message.
- Do not infer commit contents from memory, `git diff` alone, or previous discussion. If `git diff --stat` is empty but `git status -sb` shows `M  <file>`, the changes are staged.
- Leave unrelated untracked files and unrelated unstaged edits alone unless the user explicitly asks to include them.
- Do not use broad `git add .` when unrelated untracked files exist. Stage explicit paths.
- After pre-commit, re-check both staged and unstaged diffs. If hooks changed files, stage only the intended paths and rerun pre-commit until clean.
- If `git commit` fails with `.git/index.lock: Operation not permitted` in a sandboxed environment, do not restage or modify files. Rerun only `git commit -F /tmp/commitmsg.txt` with the required sandbox escalation, then remove the temp file after success.

## Commit message structure

### 1. Subject line (first line)

```
<type>[optional scope]: <description>
```

Use a scope when it helps identify the affected area. Both `fix: handle empty batches` and `fix(rollout): handle empty batches` are valid. Prefer a concise, imperative subject; about 50 characters is a writing guideline, not a hard limit. Use the same format for PR titles, which become squash commit subjects.

Project types are `feat`, `fix`, `docs`, `style`, `perf`, `refactor`, `test`, `ci`, `build`, and `chore`.

### 2. Blank line

Separate the subject from an optional body or footer with a blank line.

### 3. Body for non-trivial commits

Include a body for non-trivial commits to explain the motivation, design choices, or validation. Plain paragraphs or bullet points are sufficient. Prefer Markdown headings when they help organize a longer body; emoji are optional formatting examples.

Choose optional emoji headings using [Gitmoji](https://gitmoji.dev/) meanings. The table gives examples: 🎨 for formatting, 👷 for CI, 🔧 for configuration, and 📦️ for package changes; dependency upgrades can use ⬆️.

**Optional type-to-emoji examples:**

| Type     | Emoji | Heading Format              |
| -------- | ----- | --------------------------- |
| feat     | ✨    | `# ✨ Feature`              |
| fix      | 🐛    | `# 🐛 Bug Fix`              |
| refactor | ♻️    | `# ♻️ Refactor`             |
| perf     | ⚡️   | `# ⚡️ Performance`         |
| test     | ✅    | `# ✅ Tests`                |
| docs     | 📝    | `# 📝 Documentation`        |
| ci       | 👷    | `# 👷 CI/CD`                |
| chore    | 🔧    | `# 🔧 Chore`                |
| style    | 🎨    | `# 🎨 Style`                |
| build    | 📦️   | `# 📦️ Build`               |

**Multi-type commits**: Use the primary type in the subject. A longer body can group related details by change type.

**Multi-type body example:**

```markdown
# 🐛 Bug Fix

## Fix token expiration check

- Fix token expiration check that always returned false

---

# ♻️ Refactor

## Simplify validation logic

- Refactor auth middleware to use early return pattern

---

# ✅ Tests

## Add auth unit tests

- Add unit tests for edge cases
- Test token refresh flow
```

**Single-type body example:**

```markdown
# ✨ Feature

## Add user endpoints

- Implement GET /users/{id} endpoint
- Add POST /users for creating new users
- Include input validation middleware
```

## Full example

```bash
printf 'feat(skills): add code-review skill with checklists

# ✨ Feature

## Add code-review skill with reference documentation

- SOLID principles checklist
- Python/ML best practices
- Security review guidelines
' > /tmp/commitmsg.txt
git commit -F /tmp/commitmsg.txt
rm /tmp/commitmsg.txt
```

> **Note:** HEREDOC with `git commit -m` can fail with emoji/unicode.
> Prefer `printf ... > /tmp/commitmsg.txt`, `git commit -F /tmp/commitmsg.txt`, and `rm /tmp/commitmsg.txt` as separate commands so failures are easy to recover from.

## Important rules

- **ALWAYS** run `pre-commit run --all-files --show-diff-on-failure` before `git commit`, then `git add` again to stage any auto-fixed changes
- **ALWAYS** verify staged contents with `git diff --cached --stat` before committing
- Prefer a kebab-case scope in parentheses when it adds useful context
- **ALWAYS** use present tense imperative verb for the subject
- Include a body for non-trivial commits; prefer Markdown headings when they improve readability
- **ALWAYS** prefer `git commit -F <tmpfile>` for commits with markdown body
- **NEVER** stage unrelated untracked files or unstaged edits while creating a commit
- **NEVER** end subject with a period
- Prefer a concise subject without shortening it at the expense of clarity
- **NEVER** use generic messages ("update code", "fix bug", "changes")
- By default, only create local commits. When the user authorizes pushing or creating a PR, follow [Relax GitHub workflow](../relax-github-workflow/SKILL.md) for publication.
