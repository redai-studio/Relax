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

const mentions = (users) => users.map((user) => `@${user}`).join(', ') || '—'
const cell = (value) => String(value).replaceAll('|', '\\|').replaceAll('\n', ' ')
const row = (name, status, detail) => `| ${cell(name)} | ${cell(status)} | ${cell(detail)} |`

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
  const rows = [
    row(
      'GitHub merge state / 合入状态',
      `\`${state.mergeStateStatus}\``,
      pr.draft
        ? 'Mark ready for review / 先转为 Ready for review'
        : state.mergeStateStatus === 'CLEAN'
          ? 'GitHub reports ready / GitHub 当前报告可合入'
          : 'See the PR merge box / 查看 PR 合入区域'
    ),
  ]
  rows.push(
    row(
      'Conflicts / 冲突',
      state.mergeable === 'MERGEABLE'
        ? '✅ None / 无'
        : state.mergeable === 'CONFLICTING'
          ? '❌ Conflicts / 有冲突'
          : '⏳ Computing / 计算中',
      ''
    )
  )
  const reviewRules = rules
    .filter((rule) => rule.type === 'pull_request')
    .map((rule) => rule.parameters)
  const count = Math.max(0, ...reviewRules.map((rule) => rule.required_approving_review_count || 0))
  if (count)
    rows.push(
      row(
        'Approvals / 审批数',
        `${approved.length >= count ? '✅' : '⏳'} ${approved.length}/${count}`,
        mentions(approved)
      )
    )
  if (requested.length)
    rows.push(
      row(
        'Changes requested / 要求修改',
        '❌',
        `Address feedback and request re-review / 处理意见后请重新审批：${mentions(requested)}`
      )
    )
  for (const rule of reviewRules) {
    for (const entry of rule.required_reviewers || []) {
      const team = teams.get(entry.reviewer.id)
      const label = team
        ? `@${pr.base.repo.owner.login}/${team.slug}`
        : `${entry.reviewer.type} ${entry.reviewer.id}`
      const members = team?.members?.filter(
        (login) => login.toLowerCase() !== pr.user.login.toLowerCase()
      )
      const received = approved.filter((login) =>
        members?.some((member) => member.toLowerCase() === login.toLowerCase())
      )
      // Only an unconditional wildcard can be evaluated without reproducing GitHub's path matcher.
      const unconditional =
        entry.file_patterns?.length === 1 && ['*', '**', '**/*'].includes(entry.file_patterns[0])
      const status = !entry.minimum_approvals
        ? 'ℹ️ Optional / 可选'
        : !unconditional || !members
          ? '❔ GitHub decides / GitHub 判定'
          : `${received.length >= entry.minimum_approvals ? '✅' : '⏳'} ${received.length}/${entry.minimum_approvals}`
      const candidates = members?.filter(
        (login) => !received.some((user) => user.toLowerCase() === login.toLowerCase())
      )
      const details = []
      if (received.length) details.push(`Approved / 已审批：${mentions(received)}`)
      if (received.length < entry.minimum_approvals)
        details.push(
          candidates?.length
            ? `Ask / 可联系：${mentions(candidates)}`
            : members
              ? 'No eligible candidates / 无可审批成员'
              : 'Members unavailable / 无法读取成员'
        )
      if (!unconditional)
        details.push(`Paths / 文件条件：${(entry.file_patterns || []).join(', ')}`)
      const detail = details.join('. ')
      rows.push(row(`Team approval / 团队审批 ${label}`, status, detail))
    }
  }
  if (reviewRules.some((rule) => rule.require_last_push_approval))
    rows.push(
      row(
        'Latest push approval / 最近推送审批',
        '❔ GitHub decides / GitHub 判定',
        'Someone other than the last pusher must approve / 需由最近推送者以外的人 Approve'
      )
    )
  if (reviewRules.length)
    rows.push(
      row(
        'Code Owners',
        reviewRules.some((rule) => rule.require_code_owner_review)
          ? '❔ GitHub decides / GitHub 判定'
          : 'ℹ️ Not required / 未要求',
        ''
      )
    )
  if (unresolved !== undefined)
    rows.push(
      row(
        'Conversations / Review 对话',
        unresolved ? `⏳ ${unresolved} unresolved / 未解决` : '✅ Resolved / 已解决',
        ''
      )
    )
  const requiredChecks = [
    ...new Map(
      rules
        .flatMap((rule) =>
          rule.type === 'required_status_checks' ? rule.parameters.required_status_checks : []
        )
        .map((check) => [`${check.context}:${check.integration_id ?? 'any'}`, check])
    ).values(),
  ]
  const results = requiredChecks.map((check) => ({ ...check, status: checkStatus(check, checks) }))
  if (results.length) {
    const passed = results.filter((check) => check.status.startsWith('✅')).length
    rows.push(
      row(
        'Required CI / 必需检查',
        `${passed === results.length ? '✅' : '⏳'} ${passed}/${results.length}`,
        'Details below / 详情见下方'
      )
    )
  }
  if (rules.some((rule) => rule.parameters?.strict_required_status_checks_policy))
    rows.push(
      row(
        'Up to date / 分支更新',
        state.mergeStateStatus === 'BEHIND'
          ? '⏳ Update branch / 需要更新'
          : '❔ GitHub decides / GitHub 判定',
        'Must be up to date with base / 需要同步目标分支'
      )
    )
  const external = rules.filter(
    (rule) =>
      ![
        'pull_request',
        'required_status_checks',
        'deletion',
        'non_fast_forward',
        'creation',
        'required_linear_history',
      ].includes(rule.type)
  )
  for (const rule of external)
    rows.push(
      row(
        rule.type === 'code_quality' ? 'Code Quality' : rule.type,
        '❔ GitHub decides / GitHub 判定',
        rule.parameters?.severity || 'See repository rules / 查看仓库规则'
      )
    )
  const methods = reviewRules.map((rule) => rule.allowed_merge_methods).filter(Boolean)
  if (methods.length)
    rows.push(
      row(
        'Merge method / 合入方式',
        'ℹ️',
        methods
          .reduce((allowed, current) => allowed.filter((method) => current.includes(method)))
          .join(' / ')
      )
    )
  if (rules.some((rule) => rule.type === 'required_linear_history'))
    rows.push(row('Linear history / 线性历史', 'ℹ️', 'Squash / Rebase'))
  const lines = [
    '### Merge readiness / 合入条件',
    '',
    `Base / 目标分支：\`${cell(pr.base.ref)}\` · Head：\`${pr.head.sha.slice(0, 7)}\``,
    '',
    '| Requirement / 条件 | Status / 状态 | Next step / 下一步 |',
    '| --- | --- | --- |',
    ...rows,
    '',
    'Rules and reviews are read live; final eligibility and bypass permissions are determined by GitHub. / 根据当前规则和审批生成，最终合入资格及绕过权限以 GitHub 为准。',
  ]
  if (results.length)
    lines.push(
      '',
      '<details>',
      '<summary>Required CI / 必需检查详情</summary>',
      '',
      '| Check / 检查 | Status / 状态 |',
      '| --- | --- |',
      ...results.map((check) => `| ${cell(check.context)} | ${cell(check.status)} |`),
      '',
      '</details>'
    )
  if (warnings.length) lines.push('', ...warnings.map((warning) => `- ❔ ${cell(warning)}`))
  return lines.join('\n')
}
