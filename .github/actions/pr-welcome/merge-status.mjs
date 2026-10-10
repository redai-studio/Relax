const query = `query($owner: String!, $repo: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      headRefOid baseRefName mergeable mergeStateStatus reviewDecision
      potentialMergeCommit { oid }
      latestOpinionatedReviews(first: 100, after: $cursor) {
        nodes { author { login } authorCanPushToRepository state }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}`

const names = (users) => users.map((user) => `\`${cell(user).replaceAll('`', '\\`')}\``).join(', ')
const cell = (value) => String(value).replaceAll('|', '\\|').replaceAll('\n', ' ')
const row = (name, status) => `| ${cell(name)} | ${cell(status)} |`

async function optional(request, warnings, label, missingIsEmpty = false) {
  try {
    return await request()
  } catch (error) {
    if (missingIsEmpty && error.status === 404) return null
    warnings.push(`${label}: API unavailable / 无法读取 (${error.status || 'error'})`)
    return null
  }
}

export async function loadMergeStatus({ github, repo, number }) {
  const { data: pr } = await github.rest.pulls.get({ ...repo, pull_number: number })
  const warnings = []
  const [rules, protection] = await Promise.all([
    optional(
      () =>
        github.paginate('GET /repos/{owner}/{repo}/rules/branches/{branch}', {
          ...repo,
          branch: pr.base.ref,
          per_page: 100,
        }),
      warnings,
      'Rulesets'
    ),
    optional(
      () =>
        github.rest.repos
          .getBranchProtection({ ...repo, branch: pr.base.ref })
          .then(({ data }) => data),
      warnings,
      'Branch protection',
      true
    ),
  ])
  const activeRules = [...(rules || [])]
  if (protection?.required_pull_request_reviews) {
    activeRules.push({
      type: 'pull_request',
      parameters: {
        ...protection.required_pull_request_reviews,
        require_code_owner_review:
          protection.required_pull_request_reviews.require_code_owner_reviews,
        required_review_thread_resolution: protection.required_conversation_resolution?.enabled,
      },
    })
  } else if (protection?.required_conversation_resolution?.enabled) {
    activeRules.push({
      type: 'pull_request',
      parameters: { required_review_thread_resolution: true },
    })
  }
  if (protection?.required_status_checks) {
    const checks =
      protection.required_status_checks.checks ||
      protection.required_status_checks.contexts.map((context) => ({ context }))
    activeRules.push({
      type: 'required_status_checks',
      parameters: {
        required_status_checks: checks.map((check) => ({
          context: check.context,
          integration_id: check.app_id,
        })),
        strict_required_status_checks_policy: protection.required_status_checks.strict,
      },
    })
  }
  if (protection?.required_signatures?.enabled) activeRules.push({ type: 'required_signatures' })
  if (protection?.required_linear_history?.enabled)
    activeRules.push({ type: 'required_linear_history' })
  const reviews = []
  let state, cursor
  do {
    const result = await github.graphql(query, { ...repo, number, cursor })
    state = result.repository.pullRequest
    if (state.headRefOid !== pr.head.sha || state.baseRefName !== pr.base.ref)
      throw new Error('PR changed while collecting reviews.')
    const page = state.latestOpinionatedReviews
    reviews.push(...page.nodes)
    cursor = page.pageInfo.hasNextPage ? page.pageInfo.endCursor : null
  } while (cursor)
  const refs = [...new Set([pr.head.sha, state.potentialMergeCommit?.oid].filter(Boolean))]
  const checks = await Promise.all(
    refs.map(async (ref) => ({
      ref,
      runs: await optional(
        () =>
          github.paginate(github.rest.checks.listForRef, {
            ...repo,
            ref,
            filter: 'latest',
            per_page: 100,
          }),
        warnings,
        `Checks ${ref.slice(0, 7)}`
      ),
      statuses: await optional(
        () =>
          github.paginate(github.rest.repos.listCommitStatusesForRef, {
            ...repo,
            ref,
            per_page: 100,
          }),
        warnings,
        `Statuses ${ref.slice(0, 7)}`
      ),
    }))
  )
  const teamIds = [
    ...new Set(
      activeRules
        .flatMap((rule) => rule.parameters?.required_reviewers || [])
        .filter((entry) => entry.reviewer.type === 'Team')
        .map((entry) => entry.reviewer.id)
    ),
  ]
  const teams = new Map()
  if (teamIds.length) {
    const available = await optional(
      () => github.paginate(github.rest.teams.list, { org: repo.owner, per_page: 100 }),
      warnings,
      'Teams (read:org / Members read)'
    )
    for (const id of teamIds) {
      const team = available?.find((item) => item.id === id)
      if (!team) {
        warnings.push(`Team ${id}: membership unavailable / 无法读取成员`)
        continue
      }
      const members = await optional(
        () =>
          github.paginate(github.rest.teams.listMembersInOrg, {
            org: repo.owner,
            team_slug: team.slug,
            per_page: 100,
          }),
        warnings,
        `Team ${team.slug}`
      )
      teams.set(id, { ...team, members: members?.map((user) => user.login) })
    }
  }
  let unresolved
  if (activeRules.some((rule) => rule.parameters?.required_review_thread_resolution)) {
    let after
    unresolved = 0
    do {
      const data = await github.graphql(
        `query($owner: String!, $repo: String!, $number: Int!, $after: String) {
        repository(owner: $owner, name: $repo) { pullRequest(number: $number) {
          reviewThreads(first: 100, after: $after) { nodes { isResolved } pageInfo { hasNextPage endCursor } }
        } }
      }`,
        { ...repo, number, after }
      )
      const page = data.repository.pullRequest.reviewThreads
      unresolved += page.nodes.filter((thread) => !thread.isResolved).length
      after = page.pageInfo.hasNextPage ? page.pageInfo.endCursor : null
    } while (after)
  }
  return { pr, state, rules: activeRules, reviews, checks, teams, unresolved, warnings }
}

export function checkStatus(required, checks) {
  // GitHub prefers checks on the test merge commit when that context is present there.
  for (const source of [...checks].reverse()) {
    const runs = (source.runs || []).filter(
      (check) =>
        check.name === required.context &&
        (required.integration_id == null ||
          required.integration_id === -1 ||
          check.app?.id === required.integration_id)
    )
    runs.sort((a, b) => b.id - a.id)
    const check = runs[0]
    const states = []
    if (check) {
      states.push(
        check.status !== 'completed'
          ? '⏳ Pending / 进行中'
          : ['success', 'neutral', 'skipped'].includes(check.conclusion)
            ? '✅ Passed / 通过'
            : `❌ ${check.conclusion}`
      )
    }
    const status = (source.statuses || []).find((item) => item.context === required.context)
    if (status) {
      // Commit statuses do not expose an app ID, so an app-bound requirement is unknown.
      states.push(
        required.integration_id != null && required.integration_id !== -1
          ? '❔ Source unverified / 来源待确认'
          : status.state === 'success'
            ? '✅ Passed / 通过'
            : status.state === 'pending'
              ? '⏳ Pending / 进行中'
              : `❌ ${status.state}`
      )
    }
    if (source.runs === null || source.statuses === null)
      states.push('❔ API unavailable / 无法读取')
    if (states.length)
      return (
        states.find((value) => value.startsWith('❌')) ||
        states.find((value) => !value.startsWith('✅')) ||
        states[0]
      )
  }
  return '⏳ Not reported / 尚未上报'
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
}) {
  const currentReviews = reviews.filter(
    (review) =>
      review.author?.login &&
      review.authorCanPushToRepository &&
      review.author.login.toLowerCase() !== pr.user.login.toLowerCase()
  )
  const approved = currentReviews
    .filter((review) => review.state === 'APPROVED')
    .map((review) => review.author.login)
  const requested = currentReviews
    .filter((review) => review.state === 'CHANGES_REQUESTED')
    .map((review) => review.author.login)
  const rows = []
  const reviewRules = rules
    .filter((rule) => rule.type === 'pull_request')
    .map((rule) => rule.parameters)
  const count = Math.max(0, ...reviewRules.map((rule) => rule.required_approving_review_count || 0))
  if (count)
    rows.push(
      row(
        '审批',
        `${approved.length >= count ? '✅' : '⏳'} ${approved.length}/${count}${approved.length ? ` · 已审批：${names(approved)}` : ''}`
      )
    )
  if (requested.length) rows.push(row('修改意见', `❌ 处理意见后请重新审批：${names(requested)}`))
  const requiredTeams = new Set()
  for (const rule of reviewRules) {
    for (const entry of rule.required_reviewers || []) {
      if (!entry.minimum_approvals) continue
      const key = JSON.stringify(entry)
      if (requiredTeams.has(key)) continue
      requiredTeams.add(key)
      const team = teams.get(entry.reviewer.id)
      const label = team
        ? `团队 ${names([team.slug])}`
        : `${entry.reviewer.type} ${entry.reviewer.id}`
      const members = team?.members?.filter(
        (login) => login.toLowerCase() !== pr.user.login.toLowerCase()
      )
      const received = approved.filter((login) =>
        members?.some((member) => member.toLowerCase() === login.toLowerCase())
      )
      // Conditional paths stay unknown instead of approximating GitHub's matcher.
      const unconditional =
        entry.file_patterns?.length === 1 && ['*', '**', '**/*'].includes(entry.file_patterns[0])
      const parts = [
        !unconditional || !members
          ? '❔ GitHub 判定'
          : `${received.length >= entry.minimum_approvals ? '✅' : '⏳'} ${received.length}/${entry.minimum_approvals}`,
      ]
      if (received.length) parts.push(`已审批：${names(received)}`)
      if (received.length < entry.minimum_approvals) {
        const candidates = members?.filter(
          (login) => !received.some((user) => user.toLowerCase() === login.toLowerCase())
        )
        parts.push(
          candidates?.length
            ? `可联系：${names(candidates)}`
            : members
              ? '无可审批成员'
              : '无法读取成员'
        )
      }
      if (!unconditional) parts.push(`文件条件：${names(entry.file_patterns || [])}`)
      rows.push(row(label, parts.join(' · ')))
    }
  }
  if (reviewRules.some((rule) => rule.require_last_push_approval))
    rows.push(row('最近一次推送审批', '❔ 需由推送者以外的人 Approve，GitHub 判定'))
  if (reviewRules.some((rule) => rule.require_code_owner_review))
    rows.push(row('Code Owners', '❔ GitHub 判定'))
  if (
    reviewRules.some((rule) => rule.require_extra_approval_for_unattributed_changes) &&
    pr.user.type !== 'User'
  )
    rows.push(row('Copilot 额外审批', '❔ 无归属的 Copilot PR 需额外审批，GitHub 判定'))
  if (unresolved !== undefined)
    rows.push(row('Review 对话', unresolved ? `⏳ ${unresolved} 条未解决` : '✅ 已解决'))
  const requiredChecks = [
    ...new Map(
      rules
        .flatMap((rule) =>
          rule.type === 'required_status_checks' ? rule.parameters.required_status_checks : []
        )
        .map((check) => [`${check.context}:${check.integration_id ?? 'any'}`, check])
    ).values(),
  ]
  if (requiredChecks.length) {
    const results = requiredChecks.map((check) => checkStatus(check, checks))
    const status = results.some((value) => value.startsWith('❌'))
      ? '❌ 未通过'
      : results.some((value) => value.startsWith('❔'))
        ? '❔ 无法确认'
        : results.every((value) => value.startsWith('✅'))
          ? '✅ 已通过'
          : '⏳ 等待通过'
    rows.push(row('CI', status))
  }
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
  for (const type of new Set(
    rules.filter((rule) => !handled.includes(rule.type)).map((rule) => rule.type)
  ))
    rows.push(row(labels[type] || names([type]), '❔ GitHub 判定'))
  const modes = reviewRules.map((rule) => rule.allowed_merge_methods).filter(Boolean)
  if (rules.some((rule) => rule.type === 'required_linear_history'))
    modes.push(['squash', 'rebase'])
  const methods = modes.length
    ? modes.reduce((allowed, current) => allowed.filter((method) => current.includes(method)))
    : []
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
  if (modes.length)
    lines.push(
      `允许合入方式：${methods.length ? names(methods) : '❌ 规则没有共同允许的方式'}。`,
      ''
    )
  const url = pr.base.repo.html_url
  const sources = url
    ? `[查看合入状态](${pr.html_url}#partial-pull-merging) · [规则来源](${url}/rules)`
    : '最终以 GitHub 合入区域为准。'
  lines.push(sources)
  if (warnings.length) lines.push('', ...warnings.map((warning) => `- ❔ ${cell(warning)}`))
  lines.push('', '</details>')
  return lines.join('\n')
}
