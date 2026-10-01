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
    throw new Error('/cancel accepts workflow targets only. Use /help to list targets.');
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
    const matching = candidates
      .filter((run) => belongsToPR(run, pr, workflow))
      .sort((a, b) => b.run_number - a.run_number);
    // Concurrency can cancel a higher-numbered duplicate before another run starts.
    const selected =
      matching.find((run) => run.status !== 'completed') ||
      matching.find((run) => run.conclusion !== 'cancelled') ||
      matching[0];
    if (selected) runs.push(selected);
  }
  return runs;
}

function skipReason(run, command, target) {
  if (command === 'cancel') return run.status === 'completed' ? 'already completed.' : null;
  if (run.status !== 'completed') return `still ${run.status}; wait before rerunning.`;
  if (target.name === 'failed' && !['failure', 'timed_out'].includes(run.conclusion)) {
    return 'no failed run to rerun.';
  }
  return null;
}

async function findJob(github, repo, run, name) {
  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    ...repo,
    run_id: run.id,
    filter: 'latest',
    per_page: 100,
  });
  const job = jobs.find((item) => item.name === name);
  if (!job) throw new Error(`Check \`${name}\` was not found in the latest attempt.`);
  return job;
}

function operation(command, target, run, job) {
  if (command === 'cancel') {
    return {
      method: 'cancelWorkflowRun',
      params: { run_id: run.id },
      message: 'cancellation requested for the entire workflow.',
    };
  }
  if (job) {
    return {
      method: 'reRunJobForWorkflowRun',
      params: { job_id: job.id },
      message: `rerun requested for \`${job.name}\` and dependent jobs.`,
    };
  }
  return {
    method: target.name === 'failed' ? 'reRunWorkflowFailedJobs' : 'reRunWorkflow',
    params: { run_id: run.id },
    message: 'rerun requested.',
  };
}

async function changeRun({ github, repo, pr, run, command, target }) {
  const { data: latest } = await github.rest.actions.getWorkflowRun({ ...repo, run_id: run.id });
  const skipped = skipReason(latest, command, target);
  if (skipped) return skipped;
  const job = target.job ? await findJob(github, repo, latest, target.job) : null;
  const request = operation(command, target, latest, job);
  await assertHead(github, repo, pr.number, pr.head.sha);
  await github.rest.actions[request.method]({ ...repo, ...request.params });
  return request.message;
}

export async function runCommand({ github, repo, pr, config, command, args }) {
  const target = resolveTarget(args, config, command);
  const runs = await currentRuns(github, repo, pr, config, target);
  if (!runs.length)
    return `No matching PR workflow run exists for the current head. /${command} requires an existing run.`;
  const messages = [];
  for (const run of runs) {
    const label = `[${run.name} #${run.run_number}](${run.html_url})`;
    try {
      const message = await changeRun({ github, repo, pr, run, command, target });
      messages.push(`${label}: ${message}`);
    } catch (error) {
      // A failed response can follow a successful POST. Never automatically repeat a write.
      messages.push(
        `${label}: ${error.message.replace(/\.+$/, '')}. Inspect the run before retrying.`,
      );
    }
  }
  return `Head: \`${pr.head.sha}\`\n\n${messages.map((message) => `- ${message}`).join('\n')}`;
}
