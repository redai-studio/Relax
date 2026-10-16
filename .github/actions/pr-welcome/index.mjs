import { readFileSync } from 'node:fs'
import { loadMergeStatus, renderMergeStatus } from './merge-status.mjs'

const marker = '<!-- relax-ci:welcome -->'

export function renderWelcome({ author, repositoryUrl, defaultBranch, docsUrl, mergeStatus }) {
  const template = readFileSync(new URL('./comment.md', import.meta.url), 'utf8')
  return `${marker}\n${template
    .replaceAll('{{author}}', author)
    .replaceAll('{{docs}}', docsUrl.replace(/\/$/, ''))
    .replaceAll('{{mergeStatus}}', mergeStatus)
    .replaceAll('{{source}}', `${repositoryUrl}/tree/${encodeURIComponent(defaultBranch)}`)}`
}

export async function pullRequestNumbers({ github, context }) {
  if (context.eventName === 'pull_request_target') return [context.payload.pull_request.number]
  const run = context.payload.workflow_run
  if (!run || !['pull_request', 'pull_request_review'].includes(run.event)) return []
  const prs = await github.paginate(github.rest.pulls.list, {
    ...context.repo,
    state: 'open',
    per_page: 100,
  })
  return prs
    .filter(
      (pr) =>
        run.pull_requests?.some((associated) => associated.number === pr.number) ||
        pr.head.sha === run.head_sha ||
        pr.merge_commit_sha === run.head_sha ||
        (pr.head.ref === run.head_branch &&
          pr.head.repo?.full_name === run.head_repository?.full_name)
    )
    .map((pr) => pr.number)
}

export async function run({ github, context, botLogin, docsUrl, number }) {
  if (!botLogin) throw new Error('Configure WELCOME_BOT_LOGIN before posting welcome comments.')
  if (!Number.isSafeInteger(number) || number < 1) throw new Error('Invalid PR number.')
  const { data: identity } = await github.rest.users.getAuthenticated()
  if (identity.login.toLowerCase() !== botLogin.toLowerCase()) {
    throw new Error('WELCOME_BOT_TOKEN must belong to WELCOME_BOT_LOGIN.')
  }
  const snapshot = await loadMergeStatus({ github, repo: context.repo, number })
  if (snapshot.pr.state !== 'open') return
  const body = renderWelcome({
    author: snapshot.pr.user.login,
    repositoryUrl: context.payload.repository.html_url,
    defaultBranch: context.payload.repository.default_branch,
    docsUrl,
    mergeStatus: renderMergeStatus(snapshot),
  })
  let existing
  for await (const { data: comments } of github.paginate.iterator(github.rest.issues.listComments, {
    ...context.repo,
    issue_number: number,
    per_page: 100,
  })) {
    existing = comments.find(
      (comment) => comment.user.login === identity.login && comment.body?.startsWith(marker)
    )
    if (existing) break
  }
  if (existing?.body === body) return
  // Avoid publishing a snapshot collected before a concurrent push or base change.
  const { data: current } = await github.rest.pulls.get({ ...context.repo, pull_number: number })
  if (
    current.state !== 'open' ||
    current.head.sha !== snapshot.pr.head.sha ||
    current.base.ref !== snapshot.pr.base.ref
  ) {
    throw new Error('PR changed while collecting merge status; a later event will refresh it.')
  }
  if (existing) {
    await github.rest.issues.updateComment({ ...context.repo, comment_id: existing.id, body })
    return
  }
  await github.rest.issues.createComment({
    ...context.repo,
    issue_number: number,
    body,
  })
}
