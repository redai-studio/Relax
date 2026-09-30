const test = require('node:test');
const assert = require('node:assert/strict');
const config = require('../config.json');
const {parseCommand, authorize} = require('../commands.js');
const {assertHead} = require('../comments.js');
const {control} = require('../control.js');

test('only exact standalone registered commands are accepted', () => {
  assert.deepEqual(parseCommand(' /help\n', config), {name: 'help', args: ''});
  for (const body
           of ['please /help', '> /help', '```\n/help\n```', '/help\n/rerun', '/ci', '/re-run']) {
    assert.equal(parseCommand(body, config), null);
  }
});

test('author permissions do not grant bypass privileges', async () => {
  const input = {
    repo: {},
    pr: {user: {login: 'alice'}},
    login: 'alice',
    github: {
      rest: {repos: {getCollaboratorPermissionLevel: async () => ({data: {permission: 'read'}})}}
    }
  };
  assert.equal(await authorize({...input, permission: 'author-or-write'}), true);
  assert.equal(await authorize({...input, permission: 'maintain'}), false);
});

test('a head change prevents a mutation', async () => {
  const github = {rest: {pulls: {get: async () => ({data: {state: 'open', head: {sha: 'new'}}})}}};
  await assert.rejects(assertHead(github, {}, 1, 'old'), /head changed/);
});

test('bot accounts and ordinary issue comments are ignored without API calls', async () => {
  for (const payload
           of [{issue: {}, comment: {body: '/help'}, action: 'created'},
               {
                 issue: {pull_request: {}},
                 comment: {user: {login: config.botLogin, type: 'User'}},
                 action: 'created'
               },
  ])
    await control({context: {payload}});
});
