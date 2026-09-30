import { readFileSync } from 'node:fs';

const marker = '<!-- relax-ci:welcome -->';

export function renderWelcome({ author, repositoryUrl, defaultBranch, docsUrl }) {
  const template = readFileSync(new URL('./comment.md', import.meta.url), 'utf8');
  return `${marker}\n${template
    .replaceAll('{{author}}', author)
    .replaceAll('{{docs}}', docsUrl.replace(/\/$/, ''))
    .replaceAll('{{source}}', `${repositoryUrl}/tree/${encodeURIComponent(defaultBranch)}`)}`;
}

export async function run({ github, context, botLogin, docsUrl }) {
  if (!botLogin) throw new Error('Configure WELCOME_BOT_LOGIN before posting welcome comments.');
  const pr = context.payload.pull_request;
  const { data: identity } = await github.rest.users.getAuthenticated();
  if (identity.login.toLowerCase() !== botLogin.toLowerCase()) {
    throw new Error('WELCOME_BOT_TOKEN must belong to WELCOME_BOT_LOGIN.');
  }
  for await (const { data: comments } of github.paginate.iterator(github.rest.issues.listComments, {
    ...context.repo,
    issue_number: pr.number,
    per_page: 100,
  })) {
    if (
      comments.some(
        (comment) => comment.user.login === identity.login && comment.body.startsWith(marker),
      )
    )
      return;
  }
  await github.rest.issues.createComment({
    ...context.repo,
    issue_number: pr.number,
    body: renderWelcome({
      author: pr.user.login,
      repositoryUrl: context.payload.repository.html_url,
      defaultBranch: context.payload.repository.default_branch,
      docsUrl,
    }),
  });
}
