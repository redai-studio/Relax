async function findComment(bot, repo, number, login, marker) {
  const comments = await bot.paginate(
      bot.rest.issues.listComments, {...repo, issue_number: number, per_page: 100});
  return comments.find(comment => comment.user.login === login && comment.body.startsWith(marker));
}

async function assertHead(github, repo, number, sha) {
  const {data: latest} = await github.rest.pulls.get({...repo, pull_number: number});
  if (latest.state !== 'open' || latest.head.sha !== sha) {
    throw new Error(
        'The PR head changed or the PR closed. Submit a new command for the current head.');
  }
}

module.exports = {
  findComment,
  assertHead
};
