import config from '../../ci/config.json' with { type: 'json' }
import { runCommand } from './runs.mjs'
import { requestCherryPick } from './cherry-pick.mjs'

function parseCommand(body) {
  const firstLine = body.trim().split(/\r?\n/, 1)[0]
  const match = firstLine.match(/^\/(rerun|cancel|help|cherry-pick)(?:[ \t]+(\S+))?[ \t]*$/)
  return match ? { name: match[1], args: match[2] || '' } : null
}

function help(targets) {
  const rows = Object.entries(targets).map(
    ([name, target]) => `| \`${name}\` | ${target.job || target.workflow} |`
  )
  return [
    '## CI commands',
    '',
    '- `/rerun [failed|all|target]`: rerun failed jobs by default, or a selected workflow/check.',
    '- `/cancel <workflow|all>`: cancel entire workflows, including their matrix jobs.',
    '- `/help`: show this help.',
    '- `/review`: request Nyanpasu review.',
    '- `/cherry-pick <branch>`: label a PR merged into main for the Cherry Pick workflow (write access required).',
    '',
    'Rerun/cancel require PR authorship or repository write access. Put the command on the first line.',
    'Targets use workflow or workflow/job names, such as ci or ci/pre-commit. Job targets support rerun only.',
    '`ci` covers pre-commit checks; `all` covers every configured workflow.',
    '',
    '| Target | Check / workflow |',
    '| --- | --- |',
    ...rows,
  ].join('\n')
}

async function authorize(github, repo, pr, author) {
  if (author === pr.user.login) return
  const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
    ...repo,
    username: author,
  })
  if (!['admin', 'maintain', 'write'].includes(data.permission)) {
    throw new Error('Only the PR author or users with write access can rerun or cancel CI.')
  }
}

async function execute(github, repo, pr, author, command, cherryPickToken) {
  if (command.name === 'help') {
    if (command.args) throw new Error('Usage: /help')
    return help(config.targets)
  }
  if (command.name === 'cherry-pick') {
    return requestCherryPick({
      github,
      repo,
      pr,
      author,
      branch: command.args,
      token: cherryPickToken,
    })
  }
  await authorize(github, repo, pr, author)
  return runCommand({ github, repo, pr, config, command: command.name, args: command.args })
}

export async function run({ github, context, core, cherryPickToken }) {
  const { payload, repo } = context
  const { comment, issue } = payload
  if (!issue?.pull_request || payload.action !== 'created' || comment.user.type === 'Bot') return
  const command = parseCommand(comment.body)
  if (!command) return
  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: issue.number })
  if (pr.state !== 'open' && command.name !== 'cherry-pick') return

  const marker = `<!-- relax-ci:command:${comment.id} -->`
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...repo,
    issue_number: pr.number,
    per_page: 100,
  })
  if (comments.some((item) => item.user.type === 'Bot' && item.body.startsWith(marker))) return

  const runUrl = `${context.serverUrl}/${repo.owner}/${repo.repo}/actions/runs/${context.runId}`
  const formatReply = (message) => `${marker}\n${message}\n\n[Workflow run](${runUrl})`
  // Persist the source comment ID before any Actions write, including on workflow reruns.
  const { data: reply } = await github.rest.issues.createComment({
    ...repo,
    issue_number: pr.number,
    body: formatReply(`Processing /${command.name}.`),
  })
  let message
  try {
    message = await execute(github, repo, pr, comment.user.login, command, cherryPickToken)
  } catch (error) {
    core.warning(error.message)
    message = `Command did not complete: ${error.message}`
  }
  await github.rest.issues.updateComment({
    ...repo,
    comment_id: reply.id,
    body: formatReply(message),
  })
}
