import assert from 'node:assert/strict';
import test from 'node:test';

import { Element, Host, loadEngine } from '../krasow.mjs';

const makeHost = () => {
  const host = new Host();
  const calls = { submitted: [], written: [] };
  host.commandSet = {
    apps: {
      chat: { cancel() {}, handle() {}, mode: false, session: false },
      snake: { game: null, handleKey() {} },
    },
    execute: () => true,
  };
  host.autocomplete = {
    accept: () => false,
    clear() {},
    complete() {},
    hide() {},
    recall() {},
  };
  host.renderInput = () => {};
  host.submit = (line) => calls.submitted.push(line);
  host.write = (text, className) => calls.written.push([text, className]);
  return { calls, host };
};

test('DOM elements expose text assembled by terminal apps', () => {
  const parent = new Element('div');
  const child = new Element('span');
  child.textContent = 'world';
  parent.append('hello ', child);
  assert.equal(parent.textContent, 'hello world');

  parent.replaceChildren('new');
  assert.equal(parent.textContent, 'new');
});

test('host boots and runs commands from the website engine', async (t) => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ directories: ['/home/projects'], files: [] }),
  });
  t.after(() => globalThis.fetch = originalFetch);

  await loadEngine();
  const host = new Host();
  const output = [];
  host.write = (text) => output.push(text);
  host.renderInput = () => {};
  await host.boot();
  host.queue.push('pwd', 'echo hello', 'cd projects', 'pwd');
  await host.drain();

  assert.deepEqual(output, ['~', 'hello', '~/projects']);
});

test('commands are trimmed, recorded, capped, and routed', () => {
  const { calls, host } = makeHost();
  for (let i = 0; i < 105; i += 1) host.execute(` command-${i} `);
  assert.deepEqual(host.history.slice(0, 2), ['command-5', 'command-6']);
  assert.equal(host.history.length, 100);

  host.commandSet.execute = () => false;
  host.resolvePath = (value) => `/home/${value}`;
  host.entriesIn = (path) => (path === '/home/projects' ? [] : null);
  host.resolve = (value) => (value === 'github' ? 'https://github.com/krasow' : null);
  host.navigate = (url) => calls.navigated = url;

  host.execute('projects');
  host.execute('github');
  host.execute('missing');
  assert.deepEqual(calls.written, [
    ['zsh: is a directory: projects', 'err'],
    ['zsh: no such command, file, or directory: missing', 'err'],
  ]);
  assert.equal(calls.navigated, 'https://github.com/krasow');
});

test('line editing respects the cursor and submits the finished line', () => {
  const { calls, host } = makeHost();
  host.handleKey('a', { name: 'a' });
  host.handleKey('c', { name: 'c' });
  host.handleKey('', { name: 'left' });
  host.handleKey('b', { name: 'b' });
  assert.equal(host.ui.input.value, 'abc');

  host.handleKey('', { name: 'backspace' });
  host.handleKey('', { name: 'home' });
  host.handleKey('z', { name: 'z' });
  host.handleKey('\n', { name: 'return' });
  assert.deepEqual(calls.submitted, ['zac']);
  assert.equal(host.ui.input.value, '');
});

test('game input is translated to browser-style keys', () => {
  const { host } = makeHost();
  const keys = [];
  host.commandSet.apps.snake = {
    game: {},
    handleKey: (event) => keys.push(event.key),
  };
  host.handleKey('', { name: 'up' });
  host.handleKey('', { ctrl: true, name: 'c' });
  assert.deepEqual(keys, ['ArrowUp', 'Escape']);
});

test('home paths use shell notation', () => {
  const { host } = makeHost();
  assert.equal(host.path(), '~');
  host.currentDirectory = '/home/projects';
  assert.equal(host.path(), '~/projects');
  host.currentDirectory = '/etc';
  assert.equal(host.path(), '/etc');
});
