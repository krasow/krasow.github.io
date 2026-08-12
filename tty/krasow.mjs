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
      input: {
        value: '',
        cursor: 0,
        // SnakeGame calls focus() when a game ends — our cue to resume the shell.
        focus: () => {
          const end = this._onSnakeEnd;
          this._onSnakeEnd = null;
          end?.();
        },
      },
      autocomplete: { hidden: true, replaceChildren() {} },
      terminal: {},
    };
    this.currentDirectory = '/home';
    this.previousDirectory = null;
    this.history = [];
    this.historyCursor = 0;
    this.transcript = [];
    this.queue = [];
    this.running = false;
    this.exitRequested = false;
    this._onSnakeEnd = null;
    this._snakePre = null;
  }

  // reset re-runs `location.reload()` in the browser; here we clear in-memory
  // state (cwd, history, and the rm trash) to the same fresh-start effect.
  resetState() {
    this.currentDirectory = '/home';
    this.previousDirectory = null;
    this.history = [];
    this.historyCursor = 0;
    this.trash.paths.clear();
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
    if (this.closed) return;
    this.closed = true;
    emit(paint.cyan('→ exiting to krasow.dev'));
    process.exit(0); // the 'exit' handler restores raw mode and the cursor
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
      if (process.stdout.isTTY) process.stdout.write('\x1b[?25l');
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

  start() {
    const input = process.stdin;
    if (input.isTTY) input.setRawMode(true);
    readline.emitKeypressEvents(input);
    input.on('keypress', (str, key) => this.handleKey(str, key));
    input.on('end', () => {
      this.exitRequested = true;
      if (!this.running) this.exit();
    });
    process.on('exit', () => {
      if (input.isTTY) {
        try {
          input.setRawMode(false);
        } catch {
          // exiting anyway
        }
      }
      process.stdout.write('\x1b[?25h');
    });
    input.resume();
    this.reprompt();
  }

  reprompt() {
    this.ui.input.value = '';
    this.ui.input.cursor = 0;
    this.renderInput();
  }

  renderInput() {
    if (this.running || !process.stdout.isTTY) return;
    const prompt = this.promptText();
    const shown = color ? paint.accent(prompt) : prompt;
    process.stdout.write(`\x1b[?25h\r\x1b[K${shown} ${this.ui.input.value}`);
    process.stdout.write(`\r\x1b[${prompt.length + 1 + this.ui.input.cursor}C`);
  }

  submit(line) {
    this.queue.push(line);
    if (!this.running) this.drain();
  }

  async drain() {
    this.running = true;
    while (this.queue.length) {
      const line = this.queue.shift();
      try {
        this.execute(line);
        // A game runs itself; wait for SnakeGame to signal end via ui.input.focus().
        if (this.commandSet.apps.snake.game) await new Promise((r) => (this._onSnakeEnd = r));
        else await idle();
      } catch (error) {
        emit(`error: ${error.message}`, 'err');
      }
    }
    this.running = false;
    if (this.exitRequested) return this.exit();
    if (!this.closed) this.reprompt();
  }

  // Every key routes through the engine, exactly as terminal.js's handleKey does;
  // the host only adds the line editing a browser <input> provides for free.
  handleKey(str, key) {
    const input = this.ui.input;
    const chat = this.commandSet.apps.chat;
    const name = key?.name;

    if (this.commandSet.apps.snake.game) {
      const named = {
        up: 'ArrowUp',
        down: 'ArrowDown',
        left: 'ArrowLeft',
        right: 'ArrowRight',
        return: 'Enter',
        space: ' ',
        escape: 'Escape',
      };
      const gameKey = key?.ctrl && name === 'c' ? 'Escape' : (named[name] ?? str);
      this.commandSet.apps.snake.handleKey({ key: gameKey, preventDefault() {} });
      return;
    }

    if (key?.ctrl && name === 'c') {
      const shown = color ? paint.accent(this.promptText()) : this.promptText();
      process.stdout.write(`\r\x1b[K${shown} ${input.value}^C\n`);
      input.value = '';
      input.cursor = 0;
      this.autocomplete.hide();
      chat.cancel();
      this.reprompt();
      return;
    }
    if (key?.ctrl && name === 'd' && !input.value) return void this.exit();

    if (str === ']' && !input.value && chat.session) {
      chat.handleBracket();
      this.reprompt();
      return;
    }

    if (name === 'tab') {
      this.autocomplete.complete();
      input.cursor = input.value.length;
      return void this.renderInput();
    }
    if (name === 'up' || name === 'down') {
      this.autocomplete.recall(name === 'up' ? -1 : 1);
      input.cursor = input.value.length;
      return void this.renderInput();
    }
    if (name === 'escape') {
      this.autocomplete.clear();
      input.cursor = 0;
      return void this.renderInput();
    }

    if (name === 'return' || name === 'enter' || str === '\r' || str === '\n') {
      if (this.autocomplete.accept()) return void this.renderInput();
      const line = input.value;
      input.value = '';
      input.cursor = 0;
      this.autocomplete.hide();
      if (!this.running && process.stdout.isTTY) process.stdout.write('\n');
      this.submit(line);
      return;
    }

    if (name === 'backspace') {
      if (input.cursor > 0) {
        input.value = input.value.slice(0, input.cursor - 1) + input.value.slice(input.cursor);
        input.cursor -= 1;
      }
      this.autocomplete.hide();
      return void this.renderInput();
    }
    if (name === 'left') return void (input.cursor && (input.cursor -= 1), this.renderInput());
    if (name === 'right') {
      if (input.cursor < input.value.length) input.cursor += 1;
      return void this.renderInput();
    }
    if (name === 'home') return void ((input.cursor = 0), this.renderInput());
    if (name === 'end') return void ((input.cursor = input.value.length), this.renderInput());

    // Printable character.
    if (str && !key?.ctrl && !key?.meta && str >= ' ') {
      input.value = input.value.slice(0, input.cursor) + str + input.value.slice(input.cursor);
      input.cursor += str.length;
      this.autocomplete.hide();
      this.renderInput();
    }
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

  // reset (via the engine) calls location.reload(); re-render after clearing state.
  locationShim.reload = () => {
    host.resetState();
    if (!host.running) host.reprompt();
  };

  host.start();
};

main();
