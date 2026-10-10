const reviewQuery = `query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      headRefOid baseRefName mergeable mergeStateStatus
      potentialMergeCommit { oid }
      latestOpinionatedReviews(first: 100, after: $cursor) {
        nodes { author { login } authorCanPushToRepository state submittedAt commit { oid } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}`
const checkQuery = `query($owner: String!, $repo: String!, $number: Int!, $ref: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    object(expression: $ref) { ... on Commit {
      statusCheckRollup { contexts(first: 100, after: $cursor) {
        nodes {
          ... on CheckRun { name status conclusion isRequired(pullRequestNumber: $number) }
          ... on StatusContext { context state isRequired(pullRequestNumber: $number) }
        }
        pageInfo { hasNextPage endCursor }
      } }
    } }
  }
}`

const names = (users) => users.map((user) => `\`${user}\``).join(', ')
const cell = (value) => value.replaceAll('|', '\\|').replaceAll('\n', ' ')
const row = (name, status) => `| ${cell(name)} | ${cell(status)} |`
const sameUser = (a, b) => a.toLowerCase() === b.toLowerCase()

async function readConnection(github, query, variables, select) {
  const nodes = []
  let data, cursor
  do {
    data = await github.graphql(query, { ...variables, cursor })
    const page = select(data)
    nodes.push(...(page?.nodes || []))
    cursor = page?.pageInfo.hasNextPage ? page.pageInfo.endCursor : null
  } while (cursor)
  return { data, nodes }
}

function classicRules(protection) {
  if (!protection) return []
  const rules = []
  const reviews = protection.required_pull_request_reviews
  if (reviews || protection.required_conversation_resolution?.enabled) {
    rules.push({
      type: 'pull_request',
      parameters: {
        ...reviews,
        require_code_owner_review: reviews?.require_code_owner_reviews,
        required_review_thread_resolution: protection.required_conversation_resolution?.enabled,
      },
    })
  }
  const checks = protection.required_status_checks
  if (checks) {
    rules.push({
      type: 'required_status_checks',
      parameters: {
        required_status_checks: (
          checks.checks || checks.contexts.map((context) => ({ context }))
        ).map(({ context, app_id }) => ({ context, integration_id: app_id })),
        strict_required_status_checks_policy: checks.strict,
      },
    })
  }
  if (protection.required_signatures?.enabled) rules.push({ type: 'required_signatures' })
  return rules
}

export async function loadMergeStatus({ github, repo, number }) {
  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: number })
  const [rules, { data: protection }] = await Promise.all([
    github.paginate('GET /repos/{owner}/{repo}/rules/branches/{branch}', {
      ...repo,
      branch: pr.base.ref,
      per_page: 100,
    }),
    github.rest.repos.getBranchProtection({ ...repo, branch: pr.base.ref }).catch((error) => {
      if (error.status === 404) return { data: null }
      throw error
    }),
  ])
  rules.push(...classicRules(protection))
  const reviewRules = rules
    .filter((rule) => rule.type === 'pull_request')
    .map((rule) => rule.parameters)
  const variables = { ...repo, number }
  const { data, nodes: reviews } = await readConnection(
    github,
    reviewQuery,
    variables,
    (data) => data.repository.pullRequest.latestOpinionatedReviews
  )
  const state = data.repository.pullRequest
  if (state.headRefOid !== pr.head.sha || state.baseRefName !== pr.base.ref)
    throw new Error('PR changed while collecting reviews.')

  let lastPush
  if (
    reviewRules.some((rule) => rule.require_last_push_approval) &&
    pr.head.repo &&
    reviews.some((review) => review.state === 'APPROVED' && review.authorCanPushToRepository)
  ) {
    const headRepo = { owner: pr.head.repo.owner.login, repo: pr.head.repo.name }
    const activities = await github.paginate('GET /repos/{owner}/{repo}/activity', {
      ...headRepo,
      ref: `refs/heads/${pr.head.ref}`,
      per_page: 100,
    })
    lastPush = activities.find((activity) => activity.after === pr.head.sha)
    if (lastPush?.actor && ['push', 'branch_creation'].includes(lastPush.activity_type)) {
      const { data: changes } = await github.rest.repos.compareCommits({
        ...headRepo,
        base: lastPush.activity_type === 'branch_creation' ? pr.base.sha : lastPush.before,
        head: lastPush.after,
        per_page: 100,
      })
      // Leave pushes containing merge commits to GitHub's reviewability decision.
      lastPush.reviewable =
        changes.status === 'ahead' &&
        changes.files.length > 0 &&
        changes.commits.length === changes.total_commits &&
        changes.commits.every((commit) => commit.parents.length === 1)
    }
  }

  const refs = [...new Set([pr.head.sha, state.potentialMergeCommit?.oid].filter(Boolean))]
  const checks = await Promise.all(
    refs.map(async (ref) => {
      const { nodes } = await readConnection(
        github,
        checkQuery,
        { ...variables, ref },
        (data) => data.repository.object.statusCheckRollup?.contexts
      )
      return nodes.filter((check) => check.isRequired)
    })
  )

  const teamIds = new Set(
    reviewRules
      .flatMap((rule) => rule.required_reviewers || [])
      .filter((entry) => entry.reviewer.type === 'Team')
      .map((entry) => entry.reviewer.id)
  )
  const teams = new Map()
  const warnings = []
  if (teamIds.size) {
    try {
      const available = await github.paginate(github.rest.teams.list, {
        org: repo.owner,
        per_page: 100,
      })
      for (const team of available.filter((team) => teamIds.has(team.id))) {
        const members = await github.paginate(github.rest.teams.listMembersInOrg, {
          org: repo.owner,
          team_slug: team.slug,
          per_page: 100,
        })
        teams.set(team.id, { slug: team.slug, members: members.map((user) => user.login) })
      }
    } catch (error) {
      if (![403, 404].includes(error.status)) throw error
      warnings.push('Team 成员无法读取，请检查 read:org / Members read 权限。')
    }
  }

  let unresolved
  if (reviewRules.some((rule) => rule.required_review_thread_resolution)) {
    const { nodes } = await readConnection(
      github,
      `query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
      repository(owner: $owner, name: $repo) { pullRequest(number: $number) {
        reviewThreads(first: 100, after: $cursor) { nodes { isResolved } pageInfo { hasNextPage endCursor } }
      } }
    }`,
      variables,
      (data) => data.repository.pullRequest.reviewThreads
    )
    unresolved = nodes.filter((thread) => !thread.isResolved).length
  }
  return { pr, state, rules, reviews, checks, teams, unresolved, warnings, lastPush }
}

function ciStatus(required, checks) {
  const results = required.flatMap(({ context }) => {
    // Prefer the test merge commit when it reports this required context.
    const source =
      checks
        .toReversed()
        .find((checks) => checks.some((check) => (check.name || check.context) === context)) || []
    const reported = source.filter((check) => (check.name || check.context) === context)
    return reported.length
      ? reported.map(
          (check) => check.state || (check.status === 'COMPLETED' ? check.conclusion : 'PENDING')
        )
      : ['PENDING']
  })
  if (
    results.some(
      (result) => !['SUCCESS', 'NEUTRAL', 'SKIPPED', 'PENDING', 'EXPECTED'].includes(result)
    )
  )
    return '❌ 未通过'
  return results.every((result) => ['SUCCESS', 'NEUTRAL', 'SKIPPED'].includes(result))
    ? '✅ 已通过'
    : '⏳ 等待通过'
}

export function renderMergeStatus({
  pr,
  state,
  rules,
  reviews,
  checks,
  teams,
  unresolved,
  warnings,
  lastPush,
}) {
  const currentReviews = reviews.filter(
    (review) =>
      review.author?.login &&
      review.authorCanPushToRepository &&
      !sameUser(review.author.login, pr.user.login)
  )
  const reviewers = (state) =>
    currentReviews.filter((review) => review.state === state).map((review) => review.author.login)
  const approved = reviewers('APPROVED')
  const requested = reviewers('CHANGES_REQUESTED')
  const reviewRules = rules
    .filter((rule) => rule.type === 'pull_request')
    .map((rule) => rule.parameters)
  const enabled = (option) => reviewRules.some((rule) => rule[option])
  const rows = []
  const count = Math.max(0, ...reviewRules.map((rule) => rule.required_approving_review_count || 0))
  if (count)
    rows.push(
      row(
        `审批（至少 ${count} 位有仓库 write 或更高权限的人 Approve）`,
        `${approved.length >= count ? '✅' : '⏳'} ${approved.length}/${count}${approved.length ? ` · 已审批：${names(approved)}` : ''}`
      )
    )
  if (requested.length) rows.push(row('修改意见', `❌ 处理意见后请重新审批：${names(requested)}`))

  const requiredTeams = new Map(
    reviewRules
      .flatMap((rule) => rule.required_reviewers || [])
      .filter((entry) => entry.minimum_approvals)
      .map((entry) => [JSON.stringify(entry), entry])
  )
  for (const entry of requiredTeams.values()) {
    const team = teams.get(entry.reviewer.id)
    const label = team
      ? `团队 ${names([team.slug])}`
      : `${entry.reviewer.type} ${entry.reviewer.id}`
    const members = team?.members.filter((login) => !sameUser(login, pr.user.login))
    const received = approved.filter((login) => members?.some((member) => sameUser(member, login)))
    const candidates = members?.filter((login) => !received.some((user) => sameUser(user, login)))
    // Conditional paths stay unknown instead of approximating GitHub's matcher.
    const unconditional =
      entry.file_patterns?.length === 1 && ['*', '**', '**/*'].includes(entry.file_patterns[0])
    const parts = [
      !unconditional || !members
        ? '❔ GitHub 判定'
        : `${received.length >= entry.minimum_approvals ? '✅' : '⏳'} ${received.length}/${entry.minimum_approvals}`,
    ]
    if (received.length) parts.push(`已审批：${names(received)}`)
    if (received.length < entry.minimum_approvals)
      parts.push(
        candidates?.length
          ? `可联系：${names(candidates)}`
          : members
            ? '无可审批成员'
            : '无法读取成员'
      )
    if (!unconditional) parts.push(`文件条件：${names(entry.file_patterns || [])}`)
    rows.push(row(label, parts.join(' · ')))
  }
  if (enabled('require_last_push_approval')) {
    let status = approved.length ? '❔ GitHub 判定' : '⏳ 等待推送者以外的人 Approve'
    if (lastPush?.reviewable) {
      const received = currentReviews.filter(
        (review) =>
          review.state === 'APPROVED' &&
          review.commit?.oid === pr.head.sha &&
          review.submittedAt >= lastPush.timestamp &&
          !sameUser(review.author.login, lastPush.actor.login)
      )
      status = received.length
        ? `✅ 已由 ${names(received.map((review) => review.author.login))} Approve`
        : '⏳ 等待推送者以外的人 Approve'
    }
    rows.push(row('最近一次推送审批', status))
  }
  if (enabled('require_code_owner_review')) rows.push(row('Code Owners', '❔ GitHub 判定'))
  if (enabled('require_extra_approval_for_unattributed_changes') && pr.user.type !== 'User')
    rows.push(row('Copilot 额外审批', '❔ 无归属的 Copilot PR 需额外审批，GitHub 判定'))
  if (unresolved !== undefined)
    rows.push(row('Review 对话', unresolved ? `⏳ ${unresolved} 条未解决` : '✅ 已解决'))

  const requiredChecks = rules
    .filter((rule) => rule.type === 'required_status_checks')
    .flatMap((rule) => rule.parameters.required_status_checks)
  if (requiredChecks.length) rows.push(row('CI', ciStatus(requiredChecks, checks)))
  if (rules.some((rule) => rule.parameters?.strict_required_status_checks_policy))
    rows.push(
      row('同步目标分支', state.mergeStateStatus === 'BEHIND' ? '⏳ 需要更新' : '❔ GitHub 判定')
    )
  const handled = [
    'pull_request',
    'required_status_checks',
    'deletion',
    'non_fast_forward',
    'creation',
    'required_linear_history',
  ]
  const labels = {
    code_quality: 'Code Quality',
    code_scanning: 'Code scanning',
    required_signatures: '提交签名',
    required_deployments: '部署',
    merge_queue: '合入队列',
  }
  for (const type of new Set(rules.map((rule) => rule.type))) {
    if (!handled.includes(type)) rows.push(row(labels[type] || names([type]), '❔ GitHub 判定'))
  }
  const states = {
    CLEAN: '✅ 满足合入条件',
    BLOCKED: '⏳ 合入条件未满足',
    BEHIND: '⏳ 需要更新目标分支',
    DIRTY: '❌ 存在冲突',
    UNKNOWN: '⏳ 正在计算',
    DRAFT: '📝 Draft',
    UNSTABLE: '❔ 部分检查未通过',
    HAS_HOOKS: '❔ 等待仓库校验',
  }
  const overall = pr.draft
    ? '📝 请转为 Ready for review'
    : state.mergeable === 'CONFLICTING'
      ? '❌ 存在冲突'
      : states[state.mergeStateStatus] || names([state.mergeStateStatus])
  const lines = [
    '<details>',
    '<summary>🔎 Merge requirements / 合入条件</summary>',
    '',
    `**GitHub：${overall}**`,
    '',
  ]
  if (rows.length)
    lines.push('| Requirement / 条件 | Status / 状态 |', '| --- | --- |', ...rows, '')
  if (warnings.length) lines.push('', ...warnings.map((warning) => `- ❔ ${cell(warning)}`))
  lines.push('', '</details>')
  return lines.join('\n')
}
