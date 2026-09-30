async function assertHead(github, repo, number, sha) {
  const { data: latest } = await github.rest.pulls.get({ ...repo, pull_number: number });
  if (latest.state !== 'open' || latest.head.sha !== sha) {
    throw new Error(
      'The PR head changed or the PR closed. Submit a new command for the current head.',
    );
  }
}

function resolveTarget(args, config, command) {
  const name = args || (command === 'rerun' ? 'failed' : '');
  if (name === 'all' || (name === 'failed' && command === 'rerun')) return { name };
  const target = Object.hasOwn(config.targets, name) && config.targets[name];
  if (!target) throw new Error(`Unknown target. Use /help for /${command} syntax.`);
  if (command === 'cancel' && target.job) {
    throw new Error('/cancel accepts workflow targets only: ci, gpu-unit, integration, all.');
  }
  return { name, ...target };
}

function belongsToPR(run, pr, workflow) {
  if (
    run.event !== 'pull_request' ||
    run.head_sha !== pr.head.sha ||
    run.path !== `.github/workflows/${workflow}`
  )
    return false;
  // GitHub sometimes returns an empty pull_requests array, including for fork PRs.
  if (run.pull_requests?.length) return run.pull_requests.some((item) => item.number === pr.number);
  return (
    run.head_branch === pr.head.ref && run.head_repository?.full_name === pr.head.repo.full_name
  );
}

async function currentRuns(github, repo, pr, config, target) {
  const workflows = target.workflow
    ? [target.workflow]
    : [...new Set(Object.values(config.targets).map((item) => item.workflow))];
  const runs = [];
  for (const workflow of workflows) {
    const candidates = await github.paginate(github.rest.actions.listWorkflowRuns, {
      ...repo,
      workflow_id: workflow,
      event: 'pull_request',
      head_sha: pr.head.sha,
      per_page: 100,
    });
    const latest = candidates
      .filter((run) => belongsToPR(run, pr, workflow))
      .sort((a, b) => b.run_number - a.run_number)[0];
    if (latest) runs.push(latest);
  }
  return runs;
}

async function changeRun({ github, repo, pr, run, command, target }) {
  const label = `[${run.name} #${run.run_number}](${run.html_url})`;
  const { data: latest } = await github.rest.actions.getWorkflowRun({ ...repo, run_id: run.id });
  if (latest.head_sha !== pr.head.sha) throw new Error('Run head no longer matches the PR.');
  if (command === 'cancel') {
    if (latest.status === 'completed') return `${label}: already completed.`;
    await assertHead(github, repo, pr.number, pr.head.sha);
    await github.rest.actions.cancelWorkflowRun({ ...repo, run_id: run.id });
    return `${label}: cancellation requested for the entire workflow.`;
  }
  if (latest.status !== 'completed')
    return `${label}: still ${latest.status}; wait before rerunning.`;
  if (target.name === 'failed' && !['failure', 'timed_out'].includes(latest.conclusion)) {
    return `${label}: no failed run to rerun.`;
  }
  let job;
  if (target.job) {
    const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
      ...repo,
      run_id: run.id,
      filter: 'latest',
      per_page: 100,
    });
    job = jobs.find((item) => item.name === target.job);
    if (!job) return `${label}: check \`${target.job}\` was not found in the latest attempt.`;
    if (job.status !== 'completed') return `${label}: check is still ${job.status}.`;
  }
  await assertHead(github, repo, pr.number, pr.head.sha);
  if (job) {
    await github.rest.actions.reRunJobForWorkflowRun({ ...repo, job_id: job.id });
  } else if (target.name === 'failed') {
    await github.rest.actions.reRunWorkflowFailedJobs({ ...repo, run_id: run.id });
  } else {
    await github.rest.actions.reRunWorkflow({ ...repo, run_id: run.id });
  }
  return `${label}: rerun requested${job ? ` for \`${job.name}\` and dependent jobs` : ''}.`;
}

export async function runCommand({ github, repo, pr, config, command, args }) {
  const target = resolveTarget(args, config, command);
  const runs = await currentRuns(github, repo, pr, config, target);
  if (!runs.length)
    return 'No matching PR workflow run exists for the current head. Rerun cannot start missing workflows.';
  const messages = [];
  for (const run of runs) {
    try {
      messages.push(await changeRun({ github, repo, pr, run, command, target }));
    } catch (error) {
      // A failed response can follow a successful POST. Never automatically repeat a write.
      messages.push(
        `[${run.name}](${run.html_url}): ${error.message}. Inspect the run before retrying.`,
      );
    }
  }
  return `Head: \`${pr.head.sha}\`\n\n${messages.map((message) => `- ${message}`).join('\n')}`;
}
