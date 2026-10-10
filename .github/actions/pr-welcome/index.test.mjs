import assert from 'node:assert/strict'
import { test } from 'node:test'
import { pullRequestNumbers, renderWelcome, run } from './index.mjs'
import { checkStatus, loadMergeStatus, renderMergeStatus } from './merge-status.mjs'

const repo = { owner: 'example', repo: 'project' }
const pr = {
  number: 42,
  state: 'open',
  user: { login: 'author', type: 'User' },
  draft: false,
  head: { sha: 'a'.repeat(40), ref: 'feature', repo: { full_name: 'author/project' } },
  base: { ref: 'main', repo: { owner: { login: 'example' } } },
}
const state = {
  headRefOid: pr.head.sha,
  baseRefName: 'main',
  mergeable: 'MERGEABLE',
  mergeStateStatus: 'BLOCKED',
  reviewDecision: null,
  latestOpinionatedReviews: { nodes: [], pageInfo: {} },
}
const rules = [
  {
    type: 'pull_request',
    parameters: {
      required_approving_review_count: 1,
      require_last_push_approval: true,
      required_reviewers: [
        { reviewer: { type: 'Team', id: 1 }, minimum_approvals: 1, file_patterns: ['*'] },
      ],
    },
  },
]
const snapshot = (overrides = {}) => ({
  pr,
  state,
  rules,
  reviews: [],
  checks: [],
  teams: new Map([[1, { slug: 'reviewers', members: ['author', 'reviewer'] }]]),
  warnings: [],
  ...overrides,
})
const review = (login, status = 'APPROVED', canPush = true) => ({
  author: { login },
  state: status,
  authorCanPushToRepository: canPush,
})

test('only eligible reviews count; team approval follows approve, request changes and dismissal', () => {
  const excluded = [review('author'), review('bot', 'APPROVED', false)]
  excluded.push({ ...review('deleted'), author: null })
  const initial = renderMergeStatus(snapshot({ reviews: excluded }))
  assert.match(initial, /审批（至少 1 位有仓库 write 或更高权限的人 Approve） \| ⏳ 0\/1/)
  assert.match(initial, /最近一次推送审批 \| ⏳ 等待推送者以外的人 Approve/)
  assert.match(initial, /团队 `reviewers` \| ⏳ 0\/1 · 可联系：`reviewer`/)
  assert.doesNotMatch(initial, /可联系：.*author/)
  const approved = renderMergeStatus(snapshot({ reviews: [...excluded, review('reviewer')] }))
  assert.match(approved, /团队 `reviewers` \| ✅ 1\/1/)
  assert.doesNotMatch(approved, /可联系/)
  assert.match(approved, /最近一次推送审批 \| ❔/)
  const changed = renderMergeStatus(
    snapshot({ reviews: [review('reviewer', 'CHANGES_REQUESTED')] })
  )
  assert.match(changed, /修改意见 \| ❌/)
  assert.match(changed, /团队 `reviewers` \| ⏳ 0\/1/)
  const dismissed = renderMergeStatus(snapshot({ reviews: [review('reviewer', 'DISMISSED')] }))
  assert.match(dismissed, /团队 `reviewers` \| ⏳ 0\/1/)
})

test('unavailable members and conditional file patterns remain unknown', () => {
  assert.match(renderMergeStatus(snapshot({ teams: new Map() })), /Team 1 \| ❔ GitHub 判定/)
  const conditional = structuredClone(rules)
  conditional[0].parameters.required_reviewers[0].file_patterns = ['*', '!docs/**']
  const body = renderMergeStatus(snapshot({ rules: conditional, reviews: [review('reviewer')] }))
  assert.match(body, /团队 `reviewers` \| ❔ GitHub 判定/)
  assert.match(body, /文件条件：`\*`, `!docs\/\*\*`/)
})

test('CI is summarized and unrecognized API rules stay visible without inventing disabled requirements', () => {
  const configured = structuredClone(rules)
  configured[0].parameters.require_code_owner_review = false
  configured[0].parameters.require_extra_approval_for_unattributed_changes = true
  configured.push(
    {
      type: 'required_status_checks',
      parameters: { required_status_checks: [{ context: 'Tests' }] },
    },
    { type: 'future_merge_rule' }
  )
  const good = {
    runs: [{ name: 'Tests', status: 'completed', conclusion: 'success', id: 1 }],
    statuses: [],
  }
  const body = renderMergeStatus(snapshot({ rules: configured, checks: [good] }))
  assert.match(body, /\| CI \| ✅ 已通过 \|/)
  assert.match(body, /`future_merge_rule` \| ❔ GitHub 判定/)
  assert.doesNotMatch(body, /Code Owners|Copilot|Tests/)
  good.runs[0].conclusion = 'failure'
  assert.match(
    renderMergeStatus(snapshot({ rules: configured, checks: [good] })),
    /\| CI \| ❌ 未通过 \|/
  )
})

test('checks require the configured app and prefer test merge results', () => {
  const required = { context: 'Tests', integration_id: 7 }
  const check = (app, conclusion, id = 1) => ({
    name: 'Tests',
    app: { id: app },
    status: 'completed',
    conclusion,
    id,
  })
  const head = { runs: [check(8, 'success')], statuses: [] }
  assert.match(checkStatus(required, [head]), /Not reported/)
  head.runs.push(check(7, 'success'))
  assert.match(checkStatus(required, [head]), /Passed/)
  assert.match(
    checkStatus(required, [head, { runs: [check(7, 'failure')], statuses: [] }]),
    /failure/
  )
  assert.match(checkStatus(required, [head, { runs: [], statuses: [] }]), /Passed/)
  assert.match(
    checkStatus(required, [{ runs: [], statuses: [{ context: 'Tests', state: 'success' }] }]),
    /Source unverified/
  )
  assert.match(checkStatus(required, [{ runs: null, statuses: null }]), /API unavailable/)
  assert.match(
    checkStatus({ context: 'Tests' }, [
      { runs: [check(7, 'success')], statuses: [{ context: 'Tests', state: 'failure' }] },
    ]),
    /failure/
  )
})

test('fork workflow runs with empty associations are matched by repository and branch', async () => {
  const github = {
    rest: { pulls: { list: 'list' } },
    paginate: async () => [
      pr,
      {
        ...pr,
        number: 43,
        head: { ...pr.head, sha: 'b'.repeat(40), repo: { full_name: 'someone/project' } },
      },
    ],
  }
  const context = {
    repo,
    eventName: 'workflow_run',
    payload: {
      workflow_run: {
        event: 'pull_request_review',
        head_sha: 'old-sha',
        head_branch: 'feature',
        head_repository: { full_name: 'author/project' },
        pull_requests: [],
      },
    },
  }
  assert.deepEqual(await pullRequestNumbers({ github, context }), [42])
  context.payload.workflow_run.event = 'push'
  assert.deepEqual(await pullRequestNumbers({ github, context }), [])
})

function client({
  reviews = [],
  current = pr,
  identity = 'welcome-bot',
  comments = [],
  forbiddenTeams = false,
} = {}) {
  const writes = []
  let reads = 0
  const github = {
    rest: {
      users: { getAuthenticated: async () => ({ data: { login: identity } }) },
      pulls: { get: async () => ({ data: reads++ ? current : pr }) },
      repos: {
        getBranchProtection: async () => {
          throw Object.assign(new Error('absent'), { status: 404 })
        },
        listCommitStatusesForRef: 'statuses',
      },
      checks: { listForRef: 'checks' },
      teams: { list: 'teams', listMembersInOrg: 'members' },
      issues: {
        listComments: 'comments',
        createComment: async (body) => writes.push({ type: 'create', ...body }),
        updateComment: async (body) => writes.push({ type: 'update', ...body }),
      },
    },
    graphql: async () => ({
      repository: {
        pullRequest: { ...state, latestOpinionatedReviews: { nodes: reviews, pageInfo: {} } },
      },
    }),
    paginate: async (method) => {
      if (method.startsWith('GET')) return rules
      if (method === 'teams') {
        if (forbiddenTeams) throw Object.assign(new Error('forbidden'), { status: 403 })
        return [{ id: 1, slug: 'reviewers' }]
      }
      if (method === 'members') return ['author', 'reviewer'].map((login) => ({ login }))
      return []
    },
  }
  github.paginate.iterator = async function* () {
    yield { data: comments }
  }
  return { github, writes }
}
const context = {
  repo,
  payload: {
    repository: { html_url: 'https://github.com/example/project', default_branch: 'main' },
  },
}
const options = {
  context,
  botLogin: 'welcome-bot',
  docsUrl: 'https://example.test/docs',
  number: 42,
}

test('repeated events update the original bot comment and identical content does not write', async () => {
  const first = client()
  await run({ ...options, github: first.github })
  assert.equal(first.writes[0].type, 'create')
  assert.deepEqual(first.writes[0].body.match(/@[a-z\d-]+/gi), ['@author'])
  const comment = { id: 9, user: { login: 'welcome-bot' }, body: first.writes[0].body }
  const unchanged = client({ comments: [comment] })
  await run({ ...options, github: unchanged.github })
  assert.equal(unchanged.writes.length, 0)
  const approved = client({ comments: [comment], reviews: [review('reviewer')] })
  await run({ ...options, github: approved.github })
  assert.equal(approved.writes[0].type, 'update')
  assert.equal(approved.writes[0].comment_id, 9)
  assert.match(approved.writes[0].body, /✅ 1\/1/)
})

test('identity mismatch or a concurrent push never writes', async () => {
  const mismatch = client({ identity: 'someone' })
  await assert.rejects(run({ ...options, github: mismatch.github }), /must belong/)
  assert.equal(mismatch.writes.length, 0)
  const pushed = client({ current: { ...pr, head: { ...pr.head, sha: 'new-sha' } } })
  await assert.rejects(run({ ...options, github: pushed.github }), /PR changed/)
  assert.equal(pushed.writes.length, 0)
})

test('missing team permission is shown without breaking the welcome comment', async () => {
  const { github } = client({ forbiddenTeams: true })
  const data = await loadMergeStatus({ github, repo, number: 42 })
  assert.match(renderMergeStatus(data), /read:org \/ Members read/)
  assert.match(
    renderWelcome({
      author: 'author',
      repositoryUrl: 'https://example.test',
      defaultBranch: 'main',
      docsUrl: 'https://example.test/docs',
      mergeStatus: renderMergeStatus(data),
    }),
    /Contribution guide/
  )
})
