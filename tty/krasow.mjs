#!/usr/bin/env node
'use strict';

// krasow.dev — the website terminal, in your terminal.
//
// A host, not a reimplementation: it loads the real engine modules from
// ../terminal/js, shims the browser globals they use, and renders to a TTY.
//
//   node tty/krasow.mjs  ·  npx krasow  ·  KRASOW_BASE=http://localhost:8000 …

import readline from 'node:readline';
import { spawnSync, spawn } from 'node:child_process';
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { platform } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE = join(HERE, '..', 'terminal', 'js');
const BASE = (process.env.KRASOW_BASE || 'https://krasow.dev').replace(/\/$/, '');

const color = process.stdout.isTTY && !process.env.NO_COLOR;
const sgr = (code) => (text) => (color ? `\x1b[${code}m${text}\x1b[0m` : `${text}`);
const paint = {
  dim: sgr('2'),
  bold: sgr('1'),
  red: sgr('31'),
  green: sgr('32'),
  blue: sgr('1;34'),
  magenta: sgr('35'),
  cyan: sgr('36'),
  accent: sgr('35'),
};
const styleForClass = (className = '') =>
  className.includes('err')
    ? paint.red
    : className.includes('hint')
      ? paint.dim
      : className.includes('go')
        ? paint.cyan
        : (t) => `${t}`;
const emit = (text = '', className = '') =>
  process.stdout.write(`${styleForClass(className)(String(text))}\n`);

const colorizeEntry = (name) => {
  if (name.endsWith('/')) return paint.blue(name);
  if (name.endsWith('.sh')) return paint.green(name);
  if (/\.(pdf|vcf)$/.test(name)) return paint.magenta(name);
  if (name.endsWith('.pg')) return paint.cyan(name);
  return name;
};

// Resolve site-relative URLs against BASE; track in-flight requests so the REPL
// can wait for a command's async output before reprompting.
let pending = 0;
const fetchShim = (url, options) => {
  const full = /^https?:/.test(url) ? url : `${BASE}${url}`;
  pending += 1;
  return globalThis
    .fetch(full, { headers: { 'user-agent': 'krasow-cli' }, ...options })
    .finally(() => {
      pending -= 1;
    });
};
const idle = () =>
  new Promise((resolve) => {
    let zeros = 0;
    const timer = setInterval(() => {
      if (pending === 0) {
        if ((zeros += 1) >= 2) {
          clearInterval(timer);
          resolve();
        }
      } else zeros = 0;
    }, 20);
  });

// Minimal DOM: apps build small nodes (help spans, snake <pre>, copy textarea)
// and pass them to terminal.append(); we read their text back.
class TextNode {
  constructor(data) {
    this.data = data;
  }
  get textContent() {
    return this.data;
  }
}

class Element {
  constructor(tag) {
    this.tag = tag;
    this.className = '';
    this.childNodes = [];
    this.style = { setProperty() {} };
    this.value = '';
    this._onChange = null;
  }
  append(...items) {
    for (const item of items)
      this.childNodes.push(typeof item === 'string' ? new TextNode(item) : item);
  }
  replaceChildren(...items) {
    this.childNodes = [];
    this.append(...items);
  }
  remove() {}
  select() {}
  focus() {}
  get textContent() {
    return this.childNodes.map((node) => node.textContent).join('');
  }
  set textContent(value) {
    this.childNodes = [new TextNode(String(value))];
    this._onChange?.(String(value));
  }
}

// In-memory only: each session starts fresh, not carrying state between runs.
const localStorage = {
  _data: {},
  getItem(key) {
    return key in this._data ? this._data[key] : null;
  },
  setItem(key, value) {
    this._data[key] = String(value);
  },
  removeItem(key) {
    delete this._data[key];
  },
  clear() {
    this._data = {};
  },
};

const clipboardWrite = (text) => {
  const tools =
    platform() === 'darwin'
      ? [['pbcopy', []]]
      : platform() === 'win32'
        ? [['clip', []]]
        : [
            ['wl-copy', []],
            ['xclip', ['-selection', 'clipboard']],
            ['xsel', ['--clipboard', '--input']],
          ];
  for (const [command, args] of tools) {
    const result = spawnSync(command, args, { input: text });
    if (!result.error && result.status === 0) return true;
  }
  return false;
};

const documentShim = {
  _clipboardText: '',
  _attrs: {},
  createElement: (tag) => new Element(String(tag).toLowerCase()),
  documentElement: {
    getAttribute: (name) => documentShim._attrs[name] ?? null,
    setAttribute: (name, value) => {
      documentShim._attrs[name] = value;
    },
  },
  body: {
    append(node) {
      if (node?.tag === 'textarea') documentShim._clipboardText = node.textContent;
    },
    removeChild() {},
  },
  execCommand: (command) =>
    command === 'copy' ? clipboardWrite(documentShim._clipboardText) : false,
};

const navigatorShim = {
  clipboard: {
    writeText: async (text) => {
      if (!clipboardWrite(text)) throw new Error('no clipboard tool');
    },
  },
};

const getComputedStyleShim = () => ({
  fontSize: '14px',
  lineHeight: '17px',
  paddingTop: '0px',
  paddingBottom: '0px',
  getPropertyValue: (name) => ({ 'font-size': '14px', 'line-height': '17px' })[name] || '',
});

const locationShim = { href: '', reload() {}, assign() {}, replace() {} };

const openInBrowser = (url) => {
  const full = /^https?:/.test(url) ? url : `${BASE}${url}`;
  const command = platform() === 'darwin' ? 'open' : platform() === 'win32' ? 'start' : 'xdg-open';
  try {
    spawn(command, [full], {
      stdio: 'ignore',
      detached: true,
      shell: platform() === 'win32',
    }).unref();
  } catch {
    // no browser (headless); the printed link is the fallback
  }
  emit(paint.cyan(`→ opened ${full}`));
};

const window = {};

// Engine modules in dependency order; terminal.js/resize.js (the DOM host) excluded.
const engineOrder = (names) => {
  const core = ['app.js', 'commands.js', 'filesystem.js', 'autocomplete.js'];
  const apps = names.filter((n) => n.startsWith('apps/') || n.startsWith('games/'));
  return ['app.js', ...apps, 'commands.js', 'filesystem.js', 'autocomplete.js'].filter(
    (n) => core.includes(n) || apps.includes(n),
  );
};

// Local checkout when present, else fetch live so a shipped single file self-contains.
const readEngineSources = async () => {
  try {
    const local = [
      'app.js',
      ...readdirSync(join(ENGINE, 'apps'))
        .filter((f) => f.endsWith('.js'))
        .sort()
        .map((f) => `apps/${f}`),
      ...readdirSync(join(ENGINE, 'games'))
        .filter((f) => f.endsWith('.js'))
        .sort()
        .map((f) => `games/${f}`),
      'commands.js',
      'filesystem.js',
      'autocomplete.js',
    ];
    return engineOrder(local).map((name) => readFileSync(join(ENGINE, name), 'utf8'));
  } catch {
    const html = await (await fetchShim('/terminal/index.html')).text();
    const names = [...html.matchAll(/src="\/terminal\/js\/([^"]+\.js)"/g)]
      .map((m) => m[1])
      .filter((n) => n !== 'terminal.js' && n !== 'resize.js');
    const ordered = engineOrder(names);
    return Promise.all(
      ordered.map((name) => fetchShim(`/terminal/js/${name}`).then((r) => r.text())),
    );
  }
};

const loadEngine = async () => {
  const sources = await readEngineSources();
  const shims = {
    window,
    document: documentShim,
    localStorage,
    navigator: navigatorShim,
    getComputedStyle: getComputedStyleShim,
    location: locationShim,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    fetch: fetchShim,
    console,
    requestAnimationFrame: (fn) => setTimeout(() => fn(Date.now()), 16),
  };
  const names = Object.keys(shims);
  const values = Object.values(shims);
  for (const code of sources) {
    // eslint-disable-next-line no-new-func
    new Function(...names, code)(...values);
  }
};

// The Node host: terminal.js's interface to the engine, rendered to a TTY.
const SHORTCUTS = {
  github: 'https://github.com/krasow',
  zoom: 'https://northwestern.zoom.us/my/krasow',
};
const HISTORY_LIMIT = 100;

class Host {
  constructor() {
    const logDims = () => ({
      get clientWidth() {
        return (process.stdout.columns || 80) * 8.7;
      },
      get clientHeight() {
        return (process.stdout.rows || 24) * 17;
      },
      scrollTop: 0,
      scrollHeight: 0,
    });
    this.ui = {
      log: logDims(),
      prompt: { textContent: '' },
      input: { value: '', focus: () => this._snakeResolve?.() },
      autocomplete: { hidden: true, replaceChildren() {} },
      terminal: {},
    };
    this.currentDirectory = '/home';
    this.previousDirectory = null;
    this.history = [];
    this.historyCursor = 0;
    this.transcript = [];
    this._snakeResolve = null;
    this._snakePre = null;
  }

  async boot() {
    const DIRECTORIES = { '/': [] };
    const FILE_ROUTES = new Map();
    const TEXT_PATHS = new Set();
    const TEXT_ROUTES = new Map();
    const fs = window.KrasowTerminalFileSystem;
    await fs.loadManifest('/terminal/fs/manifest.json', {
      directories: DIRECTORIES,
      fileRoutes: FILE_ROUTES,
      textPaths: TEXT_PATHS,
      textRoutes: TEXT_ROUTES,
    });
    this.trash = new fs.VirtualTrash({
      directories: DIRECTORIES,
      fileRoutes: FILE_ROUTES,
      storageKey: 'krasow-terminal-removed-paths',
    });
    this.files = new fs.TerminalFiles(this, {
      directories: DIRECTORIES,
      fileRoutes: FILE_ROUTES,
      shortcuts: SHORTCUTS,
      textPaths: TEXT_PATHS,
      textRoutes: TEXT_ROUTES,
    });
    this.commandSet = new window.KrasowTerminalCommands.TerminalCommands(this, {
      shortcuts: SHORTCUTS,
      fileRoutes: FILE_ROUTES,
      directories: DIRECTORIES,
    });
    this.commands = this.commandSet.commands();
    this.autocomplete = new window.TerminalAutocomplete(this, {
      directories: DIRECTORIES,
      fileRoutes: FILE_ROUTES,
      textPaths: TEXT_PATHS,
    });
  }

  path() {
    return this.currentDirectory.replace(/^\/home(?=\/|$)/, '~');
  }
  promptText() {
    return this.commandSet.apps.chat.prompt(`david:${this.path()}$`);
  }
  updatePrompt() {
    this.ui.prompt.textContent = this.promptText();
  }
  persist() {} // no-op: state is in-memory
  displayFile(name) {
    return colorizeEntry(name);
  }
  write(text, className = '') {
    emit(text, className);
  }
  writeListing(entries) {
    const width = process.stdout.columns || 80;
    const columnWidth = Math.max(16, ...entries.map((n) => n.length + 3));
    const columns = Math.max(1, Math.floor(width / columnWidth));
    for (let i = 0; i < entries.length; i += columns) {
      emit(
        entries
          .slice(i, i + columns)
          .map((n) => colorizeEntry(n) + ' '.repeat(Math.max(1, columnWidth - n.length)))
          .join(''),
      );
    }
  }
  echo(command, prompt = this.promptText()) {
    emit(`${paint.accent(prompt)} ${command}`);
  }
  writeLink(url, text) {
    emit(paint.cyan(text));
  }
  clearLog() {
    process.stdout.write('\x1b[2J\x1b[H');
  }
  clearScreen() {
    process.stdout.write('\x1b[2J\x1b[H');
  }
  navigate(url) {
    if (url.endsWith('.vcf')) return this.saveFile(url);
    openInBrowser(url);
  }
  exit() {
    emit(paint.cyan('→ exiting to krasow.dev'));
    this.closed = true;
    process.exit(0);
  }
  async saveFile(url) {
    try {
      const response = await fetchShim(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const name = url.split('?')[0].split('/').pop();
      writeFileSync(join(process.cwd(), name), Buffer.from(await response.arrayBuffer()));
      emit(paint.green(`↓ saved ${name} to ${process.cwd()}`));
    } catch (error) {
      emit(`download: ${url}: ${error.message}`, 'err');
    }
  }

  append(node) {
    const className = node.className || '';
    if (className.includes('help')) return this.renderHelp(node);
    if (className.includes('snake-game')) {
      this._snakePre = node;
      node._onChange = (frame) => this.renderSnake(frame);
      return;
    }
    if (className.includes('motd')) return emit(paint.accent(node.textContent));
    emit(node.textContent, className);
  }
  renderHelp(node) {
    let command = null;
    for (const child of node.childNodes) {
      const cls = child.className || '';
      const text = child.textContent;
      if (cls.includes('help-section')) {
        emit('');
        emit(paint.bold(text));
      } else if (cls.includes('help-command')) command = text;
      else if (cls.includes('help-description')) {
        emit(`  ${paint.cyan((command || '').padEnd(26))} ${paint.dim(text)}`);
        command = null;
      } else if (cls.includes('help-note')) {
        emit('');
        emit(paint.dim(text));
      }
    }
  }
  renderSnake(frame) {
    const painted = frame
      .split('\n')
      .map((line, index) =>
        index === 0
          ? paint.dim(line)
          : line.replace(/[@o]/g, (c) => paint.green(c)).replace(/\*/g, paint.red('*')),
      )
      .join('\n');
    process.stdout.write(`\x1b[2J\x1b[H${painted}\n`);
  }

  // Mirrors terminal.js execute (readline already echoed the typed line).
  execute(raw) {
    const command = raw.trim();
    if (!command) return;
    this.history.push(command);
    this.history = this.history.slice(-HISTORY_LIMIT);
    this.historyCursor = this.history.length;

    const chat = this.commandSet.apps.chat;
    if (chat.session && !chat.mode && command === ']') return void chat.toggle();
    if (chat.mode) return void chat.handle(command);
    if (this.commandSet.execute(command)) return;
    if (this.entriesIn(this.resolvePath(command))) {
      return this.write(`zsh: is a directory: ${command}`, 'err');
    }
    const url = this.resolve(command);
    if (url) this.navigate(url);
    else this.write(`zsh: no such command, file, or directory: ${command}`, 'err');
  }

  // Raw-mode input; SnakeGame renders its own frames into the <pre>.
  runSnake(rl) {
    const input = process.stdin;
    const wasRaw = input.isRaw;
    // Detach readline's (and our Tab/`]`) keypress handlers so arrow keys drive
    // only the game, not readline history. Restored when the game ends.
    const keypressListeners = input.listeners('keypress');
    input.removeAllListeners('keypress');
    rl.pause();
    if (input.isTTY) input.setRawMode(true);
    input.resume();
    process.stdout.write('\x1b[?25l');
    const onData = (buffer) => {
      const seq = buffer.toString();
      const key =
        {
          '\x1b[A': 'ArrowUp',
          '\x1b[B': 'ArrowDown',
          '\x1b[C': 'ArrowRight',
          '\x1b[D': 'ArrowLeft',
        }[seq] ??
        (seq === '\r' || seq === '\n'
          ? 'Enter'
          : seq === '\x1b' || seq === '\x03'
            ? 'Escape'
            : seq === ' '
              ? ' '
              : seq);
      this.commandSet.apps.snake.handleKey({ key, preventDefault() {} });
    };
    return new Promise((resolve) => {
      this._snakeResolve = () => {
        input.removeListener('data', onData);
        if (input.isTTY) input.setRawMode(!!wasRaw);
        process.stdout.write('\x1b[?25h');
        if (this._snakePre) this._snakePre._onChange = null;
        this._snakeResolve = null;
        keypressListeners.forEach((listener) => input.on('keypress', listener));
        rl.resume();
        resolve();
      };
      input.on('data', onData);
    });
  }
}

const main = async () => {
  await loadEngine();
  const host = new Host();
  try {
    await host.boot();
  } catch (error) {
    emit(`could not reach ${BASE} — ${error.message}`, 'err');
    emit(paint.dim('set KRASOW_BASE to a running copy, e.g. http://localhost:8000'), 'hint');
    process.exit(1);
  }

  try {
    const motd = (await host.fileText('/etc/motd')).replace(/\s+$/, '');
    if (motd) process.stdout.write(`${paint.accent(motd)}\n\n`);
  } catch {
    // banner is optional
  }
  emit(paint.dim('# use help for more details.'));

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    completer: () => [[], ''], // Tab is handled below via the engine's autocomplete
    historySize: HISTORY_LIMIT,
  });
  const setPrompt = () =>
    rl.setPrompt(`${color ? paint.accent(host.promptText()) : host.promptText()} `);
  const reprompt = () => {
    setPrompt();
    rl.prompt();
  };
  reprompt();

  // Keystroke behaviors the web has but readline doesn't, driven by the engine's
  // own logic: Tab runs autocomplete.complete() (prefix + cycling), and `]`
  // toggles chat's shell/question mode without Enter. readline still does the
  // line editing, history, and submit.
  if (process.stdin.isTTY) {
    const setLine = (value) => {
      rl.line = value;
      rl.cursor = value.length;
      rl._refreshLine();
    };
    process.stdin.on('keypress', (str, key) => {
      if (!key || host._snakeResolve) return;
      if (key.name === 'tab') {
        host.ui.input.value = rl.line;
        host.autocomplete.complete();
        setLine(host.ui.input.value);
        return;
      }
      host.autocomplete.hide(); // any other key ends a completion cycle
      const chat = host.commandSet.apps.chat;
      if (str === ']' && chat.session && rl.line === ']') {
        rl.line = '';
        rl.cursor = 0;
        chat.handleBracket();
        setPrompt();
        rl._refreshLine();
      }
    });
  }

  // Serialize lines: for piped input readline emits every line (and `close`) up
  // front, so chain work to keep output ordered and let fetches finish first.
  let chain = Promise.resolve();
  const enqueue = (task) =>
    (chain = chain.then(task).catch((error) => emit(`error: ${error.message}`, 'err')));

  rl.on('line', (line) =>
    enqueue(async () => {
      host.execute(line);
      if (host.commandSet.apps.snake.game) await host.runSnake(rl);
      else await idle();
      if (!host.closed) reprompt();
    }),
  );

  rl.on('SIGINT', () => {
    const chat = host.commandSet.apps.chat;
    if (chat.mode || chat.confirmation) chat.cancel();
    else process.stdout.write('\n');
    reprompt();
  });

  rl.on('close', () => enqueue(async () => host.closed || host.exit()));
};

main();
