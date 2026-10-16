---
name: relax-github-workflow
description: Prepare and publish Relax GitHub contributions, create or update pull requests, and address review feedback. Use when opening a PR, choosing a contribution branch, publishing changes, following up on a review, or organizing dependent Relax PRs. Connects commit and CI workflows.
---

# Relax GitHub workflow

Use this workflow for contributions to `redai-studio/Relax`. Publishing stays within the user's request; preparing a change does not itself authorize a push, PR, or merge.

## Identify the repository and branch

Inspect the working tree, remotes, authenticated account, and target repository permissions:

```bash
git status -sb
git remote -v
gh api user --jq .login
gh api repos/redai-studio/Relax --jq '{default_branch, permissions}'
```

Suggested remote names are `upstream` for `redai-studio/Relax` and `origin` for your fork (`<your-org-name>`):

```text
upstream  https://github.com/redai-studio/Relax.git (fetch)
upstream  https://github.com/redai-studio/Relax.git (push)
origin    https://github.com/<your-org-name>/Relax.git (fetch)
origin    https://github.com/<your-org-name>/Relax.git (push)
```

Choose the remote for `redai-studio/Relax` and the push destination by their URLs, rather than assuming the remote names above; leave unrelated files and worktrees alone. For an existing PR, verify its head repository, branch and SHA before editing or publishing an update.

- Base new contributions on the current `main` of `redai-studio/Relax`, unless the task needs another base or a stack layer.
- Publish ordinary contributions from a personal fork, even with write access to `redai-studio/Relax`. Push personal branches directly to `redai-studio/Relax` only when needed, such as for shared development or gh-stack.
- Use the user's explicitly requested branch name. When none is specified and pushing directly to `redai-studio/Relax`, name the branch `<name>/<type>/<description>`. Use the contributor's chosen name or GitHub login for `<name>` and the commit type for `<type>`.
- For a fork, branch names are unrestricted; `<type>/<description>` is recommended when the user has not specified a name.

Fetch the remote for `redai-studio/Relax` before creating the branch. Use a separate worktree when it helps preserve ongoing work.

## Prepare and publish a PR

1. Implement the requested change and verify the affected behavior. Use [git-commit](../git-commit/SKILL.md) to stage only intended files, run pre-commit, and commit.
2. Read `.github/PULL_REQUEST_TEMPLATE.md`. Preserve its headings and checkboxes. Explain the background and problem, resulting behavior, and validation in enough detail for a reviewer without the task history. Link relevant PRs, issues, and public documentation; do not include internal or intranet links. Mark only checks actually completed. Use a Conventional Commits title; Relax squash merges use the PR title as the resulting commit subject.
3. When publication is authorized, push to the selected remote and create the PR with explicit repository, base and head. Prepare a body file so multiline content is passed intact:

   ```bash
   git push --set-upstream <push-remote> HEAD
   gh pr create --repo redai-studio/Relax --base <base-branch> \
       --head <head-branch-or-owner:branch> --title '<type>: <description>' --body-file <body-file>
   ```

4. Read back the PR title, body, head and base. Check the published diff before advancing review; report the PR URL and verification results. For an update, verify that the remote PR head matches the intended commit.

Use a Draft PR while work is still in progress. Mark it ready when it is ready for review. Use [relax-github-ci](../relax-github-ci/SKILL.md) to inspect required checks and investigate failures. Required checks and the repository's review requirements determine merge readiness; an optional check failure does not by itself block merging.

## Address review feedback

Read the original review thread and the affected code. Apply reasonable findings, or explain a disagreement with evidence. Verify the fix, publish the intended update, and follow up in the original thread within the user's authorization. Reuse the existing thread for the same issue and avoid repeating settled findings.

## Dependent PRs

For dependent changes, consider using `gh-stack` to split the work into cohesive, independently reviewable PRs.
