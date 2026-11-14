export async function requestCherryPick({ github, repo, pr, author, branch, token }) {
  const { data: permission } = await github.rest.repos.getCollaboratorPermissionLevel({
    ...repo,
    username: author,
  })
  if (!['admin', 'maintain', 'write'].includes(permission.permission)) {
    throw new Error('Only users with repository write access can request a cherry-pick.')
  }
  if (!pr.merged || pr.base.ref !== 'main') {
    throw new Error('/cherry-pick requires a PR already merged into main.')
  }

  const label = `cherry-pick: ${branch}`
  if (
    !/^[A-Za-z0-9][A-Za-z0-9._/-]*$/.test(branch) ||
    branch === 'HEAD' ||
    branch === pr.base.ref ||
    branch.startsWith('refs/') ||
    branch.includes('..') ||
    branch
      .split('/')
      .some(
        (part) => !part || part.startsWith('.') || part.endsWith('.') || part.endsWith('.lock')
      ) ||
    label.length > 50
  ) {
    throw new Error(
      'Usage: /cherry-pick <branch> (an existing short branch name, fitting a 50-character label).'
    )
  }
  await github.rest.repos.getBranch({ ...repo, branch })
  if (!token) throw new Error('CHERRY_PICK_BOT_TOKEN is not configured for CI Commands.')

  const labels = await github.paginate(github.rest.issues.listLabelsOnIssue, {
    ...repo,
    issue_number: pr.number,
    per_page: 100,
  })
  if (labels.some((item) => item.name === label)) {
    return `Label \`${label}\` is already present; no new cherry-pick event was requested.`
  }
  try {
    const { data: configuredLabel } = await github.rest.issues.getLabel({ ...repo, name: label })
    if (configuredLabel.name !== label) {
      throw new Error(
        `Configure the repository label \`${label}\` with exactly this capitalization.`
      )
    }
  } catch (error) {
    if (error.status !== 404) throw error
    throw new Error(`Configure the repository label \`${label}\` before requesting a cherry-pick.`)
  }

  // GITHUB_TOKEN label writes do not trigger another workflow. Only this label
  // assignment uses the bot PAT; existing CI operations and replies keep their token.
  const bot = new github.constructor({ auth: token, retry: { enabled: false } })
  await bot.rest.issues.addLabels({ ...repo, issue_number: pr.number, labels: [label] })
  return `Requested cherry-pick to \`${branch}\` via label \`${label}\`. Check the Cherry Pick workflow for the result.`
}
