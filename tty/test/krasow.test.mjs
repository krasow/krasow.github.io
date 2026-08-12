import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { Element, Host, loadEngine } from '../krasow.mjs';

const manifestResponse = (manifest) =>
  new Response(JSON.stringify(manifest), { headers: { 'content-type': 'application/json' } });

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

test('terminal output strips control sequences from remote text', () => {
  const host = new Host();
  const originalWrite = process.stdout.write;
  let output = '';
  process.stdout.write = (text) => {
    output += text;
    return true;
  };
  try {
    host.write('safe\u001b]52;c;clipboard\u0007 text');
  } finally {
    process.stdout.write = originalWrite;
  }
  assert.equal(output, 'safe�]52;c;clipboard� text\n');
});

test('host boots and runs commands from the bundled shared engine', async (t) => {
  const originalFetch = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, options) => {
    requests.push([url.href, options.redirect]);
    return manifestResponse({ directories: ['/home/projects'], files: [] });
  };
  t.after(() => (globalThis.fetch = originalFetch));

  await loadEngine();
  const host = new Host();
  const output = [];
  host.write = (text) => output.push(text);
  host.renderInput = () => {};
  await host.boot();
  host.queue.push('pwd', 'echo hello', 'cd projects', 'pwd');
  await host.drain();

  assert.deepEqual(output, ['~', 'hello', '~/projects']);
  assert.deepEqual(requests, [['https://krasow.dev/terminal/fs/manifest.json', 'error']]);
});

test('manifest data cannot introduce cross-origin fetches or navigation', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => (globalThis.fetch = originalFetch));

  for (const file of [
    { path: '/home/text.md', url: 'https://evil.example/text', text: true },
    { path: '/home/page', target: 'https://evil.example/page', text: false },
  ]) {
    globalThis.fetch = async () => manifestResponse({ files: [file] });
    await loadEngine();
    await assert.rejects(
      new Host().boot(),
      /Invalid url in manifest|cross-origin fetch blocked|navigation origin is not trusted/,
    );
  }
});

test('remote chat data cannot dispatch arbitrary or unconfirmed side effects', async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => (globalThis.fetch = originalFetch));
  globalThis.fetch = async (url) => {
    if (url.pathname.endsWith('/manifest.json')) return manifestResponse({ files: [] });
    return manifestResponse({
      fallback: 'fallback',
      entries: [{ answer: 'bad', command: 'rm -rf /home', keywords: ['bad'] }],
    });
  };

  await loadEngine();
  const host = new Host();
  await host.boot();
  await assert.rejects(host.commandSet.apps.chat.model.load(), /invalid knowledge data/);

  const chat = host.commandSet.apps.chat;
  const output = [];
  const commands = [];
  host.append = () => {};
  host.echo = () => {};
  host.write = (message, className) => output.push([message, className]);
  host.commands = new Map([['open', (args) => commands.push(args)]]);
  chat.model.ask = async () => ({ answer: 'page', command: 'open /home/about.pg' });

  await chat.ask('about');
  assert.deepEqual(commands, []);
  assert.match(output.at(-1)[0], /run `open \/home\/about\.pg` to continue/);

  chat.mode = true;
  await chat.ask('about');
  assert.deepEqual(commands, []);
  assert.equal(chat.confirmation.command, 'open /home/about.pg');
  chat.confirm('yes');
  assert.deepEqual(commands, [['/home/about.pg']]);
});

test('contact-card downloads are bounded and never overwrite files', async (t) => {
  const originalFetch = globalThis.fetch;
  const originalDirectory = process.cwd();
  const directory = mkdtempSync(join(tmpdir(), 'krasow-download-'));
  t.after(() => {
    process.chdir(originalDirectory);
    globalThis.fetch = originalFetch;
    rmSync(directory, { force: true, recursive: true });
  });
  process.chdir(directory);

  globalThis.fetch = async () => new Response('BEGIN:VCARD\nEND:VCARD\n');
  await loadEngine();
  const host = new Host();
  const output = [];
  host.write = (message, className) => output.push([message, className]);

  await host.saveFile('/contact.vcf');
  assert.equal(readFileSync(join(directory, 'contact.vcf'), 'utf8'), 'BEGIN:VCARD\nEND:VCARD\n');

  writeFileSync(join(directory, 'existing.vcf'), 'original');
  await host.saveFile('/existing.vcf');
  assert.equal(readFileSync(join(directory, 'existing.vcf'), 'utf8'), 'original');

  globalThis.fetch = async () => new Response(Buffer.alloc(1024 * 1024 + 1));
  await loadEngine();
  await host.saveFile('/large.vcf');
  assert.equal(output.at(-1)[1], 'err');
  assert.match(output.at(-1)[0], /response is too large/);
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
  host.navigate = (url) => (calls.navigated = url);

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
