const config = require('./config.json');
const {parseCommand, help, authorize} = require('./commands.js');
const {findComment, assertHead} = require('./comments.js');

async function control({github, bot, context, core, botLogin = config.botLogin}) {
  const {payload, repo} = context;
  const comment = payload.comment;
  if (!payload.issue?.pull_request || payload.action !== 'created' || !comment) return;
  if (comment.user.type === 'Bot' || comment.user.login === botLogin) return;
  const command = parseCommand(comment.body, config);
  if (!command || config.commands[command.name].external) return;
  const number = payload.issue.number;
  const {data: pr} = await github.rest.pulls.get({...repo, pull_number: number});
  if (pr.state !== 'open') return;
  const marker = `<!-- relax-ci:command:${comment.id} -->`;
  if (await findComment(bot, repo, number, botLogin, marker)) return;

  const {data: identity} = await bot.rest.users.getAuthenticated();
  if (identity.login !== botLogin)
    throw new Error('WELCOME_BOT_TOKEN must belong to WELCOME_BOT_LOGIN.');

  // Claim the event before any action. Retrying this workflow must not repeat a
  // mutation.
  const {data: reply} = await bot.rest.issues.createComment({
    ...repo,
    issue_number: number,
    body: `${marker}\nProcessing /${command.name} for \`${pr.head.sha}\`.`,
  });
  let message;
  try {
    if (!await authorize({
          github,
          repo,
          pr,
          login: comment.user.login,
          permission: config.commands[command.name].permission
        })) {
      throw new Error('You do not have permission to use this command.');
    }
    await assertHead(github, repo, number, pr.head.sha);
    if (command.name === 'help') {
      if (command.args) throw new Error('Usage: /help');
      message = help(config);
    }
  } catch (error) {
    core.warning(error.message);
    message = `Command did not complete: ${
        error.message}\n\nInspect this workflow run before submitting a new command.`;
  }
  await bot.rest.issues.updateComment(
      {...repo, comment_id: reply.id, body: `${marker}\n${message}`});
}

module.exports = {control};
