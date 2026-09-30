// Keep parsing independent of GitHub so command syntax can be tested locally.
export function parseCommand(body, config) {
  const match = body.trim().match(/^\/([a-z]+)(?:[ \t]+([^\r\n]*))?$/);
  if (!match || !Object.hasOwn(config.commands, match[1])) return null;
  return { name: match[1], args: (match[2] || '').trim() };
}

export function help(config) {
  const commands = Object.values(config.commands)
    .map(({ usage, description }) => `| \`${usage}\` | ${description} |`)
    .join('\n');
  const targets = Object.entries(config.targets)
    .map(([name, target]) => `| \`${name}\` | ${target.job || target.workflow} |`)
    .join('\n');
  return `## CI commands\n\nUse one command per PR comment.\n\n| Command | Purpose |\n| --- | --- |\n${
    commands
  }\n\n| Target | Check / workflow |\n| --- | --- |\n${targets}`;
}

export async function authorize({ github, repo, pr, login, permission }) {
  if (permission === 'any') return true;
  if (permission === 'author-or-write' && login === pr.user.login) return true;
  const { data } = await github.rest.repos.getCollaboratorPermissionLevel({
    ...repo,
    username: login,
  });
  return permission === 'maintain'
    ? data.permission === 'admin' || data.role_name === 'maintain'
    : ['admin', 'maintain', 'write'].includes(data.permission);
}
