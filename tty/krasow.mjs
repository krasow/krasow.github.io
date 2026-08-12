#!/usr/bin/env node
'use strict';

// krasow.dev — the website terminal, in your terminal.
//
// A host, not a reimplementation: it loads the bundled shared engine package,
// shims the browser globals it uses, and renders to a TTY.
//
//   node tty/krasow.mjs  ·  npx krasow  ·  KRASOW_BASE=http://localhost:8000 …

import readline from 'node:readline';
import { spawnSync, spawn } from 'node:child_process';
import { writeFileSync, realpathSync } from 'node:fs';
import { platform } from 'node:os';
import { basename, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_BASE = 'https://krasow.dev';
const MAX_DOWNLOAD_BYTES = 1024 * 1024;
const MAX_MANIFEST_BYTES = 512 * 1024;
const TRUSTED_NAVIGATION_ORIGINS = new Set([
  'https://github.com',
  'https://ieeexplore.ieee.org',
  'https://journals.sagepub.com',
  'https://julialegate.github.io',
  'https://legion.stanford.edu',
  'https://northwestern.zoom.us',
  'https://www.computer.org',
  'https://www.mccormick.northwestern.edu',
]);

const isLoopback = (hostname) =>
  hostname === 'localhost' ||
  hostname.endsWith('.localhost') ||
  hostname === '127.0.0.1' ||
  hostname === '[::1]';

const parseBase = (value) => {
  const url = new URL(value);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && isLoopback(url.hostname)))
    throw new Error('KRASOW_BASE must use HTTPS (HTTP is allowed only for localhost)');
  if (url.username || url.password) throw new Error('KRASOW_BASE must not contain credentials');
  url.hash = '';
  url.search = '';
  return url.href.replace(/\/$/, '');
};

const BASE = parseBase(process.env.KRASOW_BASE || DEFAULT_BASE);
const BASE_ORIGIN = new URL(BASE).origin;
TRUSTED_NAVIGATION_ORIGINS.add(BASE_ORIGIN);

const resolveUrl = (value) => {
  if (typeof value !== 'string' || !value) throw new Error('URL must be a non-empty string');
  if (/[\u0000-\u001f\u007f]/.test(value))
    throw new Error(`control characters are not allowed in URLs: ${value}`);
  if (value.includes('\\')) throw new Error(`backslashes are not allowed in URLs: ${value}`);
  if (value.startsWith('//')) throw new Error(`scheme-relative URL is not allowed: ${value}`);
  const url = /^[a-z][a-z\d+.-]*:/i.test(value)
    ? new URL(value)
    : new URL(`${BASE}/${value.replace(/^\//, '')}`);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error(`unsupported URL: ${value}`);
  if (url.username || url.password)
    throw new Error(`credentials are not allowed in URLs: ${value}`);
  return url;
};

const fetchUrl = (value) => {
  const url = resolveUrl(value);
  if (url.origin !== BASE_ORIGIN) throw new Error(`cross-origin fetch blocked: ${url.origin}`);
  return url;
};

const navigationUrl = (value) => {
  const url = resolveUrl(value);
  if (!TRUSTED_NAVIGATION_ORIGINS.has(url.origin))
    throw new Error(`navigation origin is not trusted: ${url.origin}`);
  return url;
};

const responseBuffer = async (response, limit) => {
  const length = Number(response.headers?.get('content-length'));
  if (Number.isFinite(length) && length > limit) throw new Error('response is too large');
  if (!response.body?.getReader) {
    const buffer = Buffer.from(await response.arrayBuffer());
    if (buffer.byteLength > limit) throw new Error('response is too large');
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let size = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > limit) {
      await reader.cancel();
      throw new Error('response is too large');
    }
    chunks.push(Buffer.from(value));
  }
  return Buffer.concat(chunks, size);
};

const manifestData = async (response, engine) => {
  const manifest = JSON.parse(
    (await responseBuffer(response, MAX_MANIFEST_BYTES)).toString('utf8'),
  );
  const validated = engine.validateManifest(manifest);
  for (const file of validated.files) {
    if (file.text ?? true) fetchUrl(file.url ?? file.target);
    else if (file.url) fetchUrl(file.url);
    if (file.target) {
      const target = resolveUrl(file.target);
      if (target.pathname.toLowerCase().endsWith('.vcf')) fetchUrl(file.target);
      else navigationUrl(file.target);
    }
  }
  return validated;
};

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
const sanitizeTerminalText = (text) =>
  String(text).replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g, '\ufffd');
const styleForClass = (className = '') =>
  className.includes('err')
    ? paint.red
    : className.includes('hint')
      ? paint.dim
      : className.includes('go')
        ? paint.cyan
        : className.includes('bold')
          ? paint.bold
          : className.includes('accent')
            ? paint.accent
            : (t) => `${t}`;
const emit = (text = '', className = '') =>
  process.stdout.write(`${styleForClass(className)(sanitizeTerminalText(text))}\n`);
const emitStyled = (text = '') => process.stdout.write(`${text}\n`);

const colorizeEntry = (name) => {
  name = sanitizeTerminalText(name);
  if (name.endsWith('/')) return paint.blue(name);
  if (name.endsWith('.sh')) return paint.green(name);
  if (/\.(pdf|vcf)$/.test(name)) return paint.magenta(name);
  if (name.endsWith('.pg')) return paint.cyan(name);
  return name;
};

// Resolve site-relative URLs against BASE; track in-flight requests so the REPL
// can wait for a command's async output before reprompting.
let pending = 0;
let networkFetch;
const fetchShim = (url, options = {}) => {
  const full = fetchUrl(url);
  const headers = new Headers(options.headers);
  if (!headers.has('user-agent')) headers.set('user-agent', 'krasow-cli');
  pending += 1;
  return networkFetch(full, {
    ...options,
    headers,
    redirect: 'error',
  }).finally(() => {
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
let store = {};
const localStorage = {
  getItem: (k) => store[k] ?? null,
  setItem: (k, v) => (store[k] = String(v)),
  removeItem: (k) => delete store[k],
  clear: () => (store = {}),
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

const getComputedStyleShim = () => ({ fontSize: '14px', lineHeight: '17px' });

const locationShim = { href: '', reload() {}, assign() {}, replace() {} };

const openInBrowser = (url) => {
  let full;
  try {
    full = navigationUrl(url).href;
  } catch (error) {
    emit(`open: ${url}: ${error.message}`, 'err');
    return;
  }
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
  emit(`→ opened ${full}`, 'go');
};

const window = {};

const loadEngine = async () => {
  if (!networkFetch || globalThis.fetch !== fetchShim)
    networkFetch = globalThis.fetch.bind(globalThis);
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
  for (const [name, value] of Object.entries(shims))
    Object.defineProperty(globalThis, name, { configurable: true, value, writable: true });
  await import('@krasow/terminal-engine/engine');
};

// The Node host: terminal.js's interface to the engine, rendered to a TTY.
const SHORTCUTS = {
  github: 'https://github.com/krasow',
  zoom: 'https://northwestern.zoom.us/my/krasow',
};
const HISTORY_LIMIT = 100;

class Host {
  constructor() {
    this.ui = {
      // cowsay/snake read these; snake also writes scrollTop.
      log: {
        get clientWidth() {
          return (process.stdout.columns || 80) * 8.7;
        },
        get clientHeight() {
          return (process.stdout.rows || 24) * 17;
        },
        scrollTop: 0,
        scrollHeight: 0,
      },
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
    };
    this.currentDirectory = '/home';
    this.previousDirectory = null;
    this.history = [];
    this.historyCursor = 0;
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
    const fs = window.KrasowTerminalFileSystem;
    const opts = {
      directories: { '/': [] },
      fileRoutes: new Map(),
      textPaths: new Set(),
      textRoutes: new Map(),
    };
    const response = await fetchShim('/terminal/fs/manifest.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const manifest = await manifestData(response, fs);
    fs.hydrate(manifest.files, opts, manifest.directories);
    this.trash = new fs.VirtualTrash({ ...opts, storageKey: 'krasow-terminal-removed-paths' });
    this.files = new fs.TerminalFiles(this, { ...opts, shortcuts: SHORTCUTS });
    this.commandSet = new window.KrasowTerminalCommands.TerminalCommands(this, {
      ...opts,
      shortcuts: SHORTCUTS,
    });
    this.commands = this.commandSet.commands();
    this.autocomplete = new window.TerminalAutocomplete(this, opts);
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
      emitStyled(
        entries
          .slice(i, i + columns)
          .map((n) => colorizeEntry(n) + ' '.repeat(Math.max(1, columnWidth - n.length)))
          .join(''),
      );
    }
  }
  echo(command, prompt = this.promptText()) {
    emitStyled(`${paint.accent(sanitizeTerminalText(prompt))} ${sanitizeTerminalText(command)}`);
  }
  writeLink(url, text) {
    emit(text, 'go');
  }
  clearScreen() {
    process.stdout.write('\x1b[2J\x1b[H');
  }
  clearLog() {
    this.clearScreen();
  }
  navigate(url) {
    let target;
    try {
      target = resolveUrl(url);
    } catch (error) {
      return void this.write(`open: ${url}: ${error.message}`, 'err');
    }
    if (target.pathname.toLowerCase().endsWith('.vcf')) return this.saveFile(url);
    openInBrowser(url);
  }
  exit() {
    if (this.closed) return;
    this.closed = true;
    emit('→ exiting to krasow.dev', 'go');
    process.exit(0); // the 'exit' handler restores raw mode and the cursor
  }
  async saveFile(url) {
    try {
      const target = fetchUrl(url);
      if (!target.pathname.toLowerCase().endsWith('.vcf'))
        throw new Error('only contact cards can be saved');
      const encodedName = basename(target.pathname);
      const name = decodeURIComponent(encodedName);
      if (
        !name ||
        name === '.' ||
        name === '..' ||
        basename(name) !== name ||
        !name.toLowerCase().endsWith('.vcf')
      )
        throw new Error('unsafe download filename');
      const response = await fetchShim(target.href);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const contents = await responseBuffer(response, MAX_DOWNLOAD_BYTES);
      writeFileSync(join(process.cwd(), name), contents, { flag: 'wx', mode: 0o600 });
      this.write(`↓ saved ${name} to ${process.cwd()}`, 'go');
    } catch (error) {
      const message =
        error.code === 'EEXIST' ? 'file already exists (not overwritten)' : error.message;
      this.write(`download: ${url}: ${message}`, 'err');
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
    if (className.includes('motd')) return emit(node.textContent, 'accent');
    emit(node.textContent, className);
  }
  renderHelp(node) {
    let command = null;
    for (const child of node.childNodes) {
      const cls = child.className || '';
      const text = child.textContent;
      if (cls.includes('help-section')) {
        emit('');
        emit(text, 'bold');
      } else if (cls.includes('help-command')) command = text;
      else if (cls.includes('help-description')) {
        emitStyled(
          `  ${paint.cyan(sanitizeTerminalText(command || '').padEnd(26))} ${paint.dim(sanitizeTerminalText(text))}`,
        );
        command = null;
      } else if (cls.includes('help-note')) {
        emit('');
        emit(text, 'hint');
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

  // Mirrors terminal.js execute; the typed line was already echoed on Enter.
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

  // The input line as a string: return to column 0, clear it, draw prompt + value.
  drawLine() {
    const p = this.promptText();
    return `\r\x1b[K${color ? paint.accent(p) : p} ${this.ui.input.value}`;
  }

  reprompt() {
    this.ui.input.value = '';
    this.ui.input.cursor = 0;
    this.renderInput();
  }

  renderInput() {
    if (this.running || !process.stdout.isTTY) return;
    const col = this.promptText().length + 1 + this.ui.input.cursor;
    process.stdout.write(`\x1b[?25h${this.drawLine()}\r\x1b[${col}C`);
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
    const { snake, chat } = this.commandSet.apps;
    const name = key?.name;
    const ctrl = key?.ctrl;

    // A running game owns every key.
    if (snake.game) {
      const named = {
        up: 'ArrowUp',
        down: 'ArrowDown',
        left: 'ArrowLeft',
        right: 'ArrowRight',
        return: 'Enter',
        space: ' ',
        escape: 'Escape',
      };
      return void snake.handleKey({
        key: ctrl && name === 'c' ? 'Escape' : (named[name] ?? str),
        preventDefault() {},
      });
    }

    if (ctrl && name === 'c') {
      process.stdout.write(`${this.drawLine()}^C\n`);
      this.autocomplete.hide();
      chat.cancel();
      return void this.reprompt();
    }
    if (ctrl && name === 'd' && !input.value) return void this.exit();
    if (str === ']' && !input.value && chat.session)
      return void (chat.handleBracket(), this.reprompt());

    if (name === 'return' || str === '\r' || str === '\n') {
      if (this.autocomplete.accept()) return void this.renderInput();
      const line = input.value;
      this.autocomplete.hide();
      input.value = '';
      input.cursor = 0;
      if (!this.running && process.stdout.isTTY) process.stdout.write('\n');
      return void this.submit(line);
    }

    // Completion/history set input.value themselves; park the cursor at the end.
    if (name === 'tab' || name === 'up' || name === 'down') {
      name === 'tab'
        ? this.autocomplete.complete()
        : this.autocomplete.recall(name === 'up' ? -1 : 1);
      input.cursor = input.value.length;
      return void this.renderInput();
    }
    if (name === 'escape')
      return void (this.autocomplete.clear(), (input.cursor = 0), this.renderInput());

    // Cursor movement.
    const move = {
      left: -1,
      right: 1,
      home: -input.cursor,
      end: input.value.length - input.cursor,
    };
    if (name in move) {
      input.cursor = Math.max(0, Math.min(input.value.length, input.cursor + move[name]));
      return void this.renderInput();
    }

    // Editing: backspace, or insert a printable character at the cursor.
    if (name === 'backspace') {
      input.value =
        input.value.slice(0, Math.max(0, input.cursor - 1)) + input.value.slice(input.cursor);
      if (input.cursor) input.cursor -= 1;
    } else if (str && !ctrl && !key?.meta && str >= ' ') {
      input.value = input.value.slice(0, input.cursor) + str + input.value.slice(input.cursor);
      input.cursor += str.length;
    } else {
      return;
    }
    this.autocomplete.hide();
    this.renderInput();
  }
}

const main = async () => {
  await loadEngine();
  const host = new Host();
  try {
    await host.boot();
  } catch (error) {
    emit(`could not reach ${BASE} — ${error.message}`, 'err');
    emit('set KRASOW_BASE to a running copy, e.g. http://localhost:8000', 'hint');
    process.exit(1);
  }

  try {
    const motd = (await host.fileText('/etc/motd')).replace(/\s+$/, '');
    if (motd) process.stdout.write(`${paint.accent(motd)}\n\n`);
  } catch {
    // banner is optional
  }
  emit('# use help for more details.', 'hint');

  // reset (via the engine) calls location.reload(); re-render after clearing state.
  locationShim.reload = () => {
    host.resetState();
    if (!host.running) host.reprompt();
  };

  host.start();
};

export { Element, Host, TextNode, loadEngine };

if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) main();
