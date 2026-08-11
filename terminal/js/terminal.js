(() => {
  'use strict';

  const SHORTCUTS = {
    github: 'https://github.com/krasow',
    zoom: 'https://northwestern.zoom.us/my/krasow',
  };
  const PAGE_SOURCES = {};
  const DIRECTORIES = {
    '/': [],
  };
  const FILE_ROUTES = new Map();
  const TEXT_PATHS = new Set();

  const STORAGE_KEY = 'krasow-terminal-state';
  const REMOVED_PATHS_KEY = 'krasow-terminal-removed-paths';
  const HISTORY_LIMIT = 10;
  const NOT_FOUND_PATH = new URLSearchParams(location.search).get('notFound');

  const byId = (id) => document.getElementById(id);
  const makeElement = (tag, className, text = '') => {
    const node = document.createElement(tag);
    node.className = className;
    node.textContent = text;
    return node;
  };
  const fileIcon = (name) => {
    if (name.endsWith('/')) return '\uf07b';
    if (name.endsWith('.pg')) return '\uf0ac';
    if (name.endsWith('.md')) return '\ue73e';
    if (name.endsWith('.pdf')) return '\uf1c1';
    if (name.endsWith('.sh')) return '\uf489';
    if (name.endsWith('.pub')) return '\uf084';
    if (name.endsWith('.vcf')) return '\uf2bb';
    return '\uf15b';
  };
  const displayFile = (name) => `${fileIcon(name)} ${name}`;

  class Terminal {
    constructor() {
      this.ui = {
        terminal: document.querySelector('.term'),
        form: byId('cmd'),
        log: byId('log'),
        input: byId('in'),
        prompt: byId('prompt'),
        autocomplete: byId('autocomplete'),
        mobileKeys: document.querySelector('.mobile-keys'),
      };

      const resizeHandle = document.querySelector('.terminal-resize');
      this.resizer = window.KrasowTerminalResize
        ? new window.KrasowTerminalResize.TerminalResizer(this.ui.terminal, resizeHandle)
        : null;
      this.trash = new window.KrasowTerminalFileSystem.VirtualTrash({
        directories: DIRECTORIES,
        fileRoutes: FILE_ROUTES,
        storageKey: REMOVED_PATHS_KEY,
      });

      this.currentDirectory = '/home';
      this.previousDirectory = null;
      this.files = new window.KrasowTerminalFileSystem.TerminalFiles(this, {
        directories: DIRECTORIES,
        fileRoutes: FILE_ROUTES,
        pageSources: PAGE_SOURCES,
        shortcuts: SHORTCUTS,
        textPaths: TEXT_PATHS,
      });
      this.history = [];
      this.historyCursor = 0;
      this.transcript = [];
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
      this.restore();
    }

    start() {
      this.ui.form.addEventListener('submit', (event) => {
        event.preventDefault();
        this.autocomplete.hide();
        this.execute(this.ui.input.value);
        this.ui.input.value = '';
      });
      this.ui.input.addEventListener('input', () => this.autocomplete.hide());
      this.ui.input.addEventListener('keydown', (event) => this.handleKey(event));
      this.ui.mobileKeys.addEventListener('click', (event) => {
        const key = event.target.closest('[data-key]')?.dataset.key;
        if (key) this.handleKey({ key, preventDefault() {} });
        this.ui.input.focus();
      });
      this.ui.terminal.addEventListener('click', () => this.ui.input.focus());
      this.resizer?.start();
      this.ui.input.focus();
    }

    execute(raw) {
      const command = raw.trim();
      if (!command) return;
      this.echo(command);

      this.history.push(command);
      this.history = this.history.slice(-HISTORY_LIMIT);
      this.historyCursor = this.history.length;
      this.persist();

      if (this.commandSet.apps.chat.session && !this.commandSet.apps.chat.mode && command === ']') {
        this.commandSet.apps.chat.toggle();
        return;
      }

      if (this.commandSet.apps.chat.mode) {
        this.commandSet.apps.chat.handle(command);
        return;
      }

      if (this.commandSet.execute(command)) return;

      if (this.entriesIn(this.resolvePath(command))) {
        this.write(`zsh: is a directory: ${command}`, 'err');
        return;
      }

      const url = this.resolve(command);
      if (url) this.navigate(url);
      else this.write(`zsh: no such command, file, or directory: ${command}`, 'err');
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

    writeListing(entries, track = true) {
      const listing = makeElement('div', 'ln pth ls-grid');
      const columnWidth = Math.max(16, ...entries.map((name) => name.length + 3));
      listing.style.setProperty('--ls-column-width', `${columnWidth}ch`);
      listing.append(...entries.map((name) => makeElement('span', 'ls-entry', displayFile(name))));
      this.append(listing, track ? { type: 'listing', entries } : null);
    }

    displayFile(name) {
      return displayFile(name);
    }

    navigate(url) {
      const download = url.endsWith('.vcf');
      this.writeLink(url, download ? '→ downloading contact.vcf' : `→ ${url}`);
      if (download)
        setTimeout(() => {
          location.href = url;
        }, 120);
      else window.open(url, '_blank', 'noopener,noreferrer');
    }

    write(text, className = '') {
      this.append(makeElement('p', `ln ${className}`.trim(), text), {
        type: 'line',
        text,
        className,
      });
    }

    writeLink(url, text) {
      const line = makeElement('p', 'ln go');
      const link = makeElement('a', '', text);
      link.href = url;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      line.append(link);
      this.append(line, { type: 'link', url, text });
    }

    echo(command, prompt = this.promptText()) {
      const line = makeElement('p', 'ln');
      line.append(makeElement('span', 'pr', prompt), ` ${command}`);
      this.append(line, { type: 'echo', prompt, command });
    }

    append(node, record = null) {
      this.ui.log.append(node);
      this.ui.log.scrollTop = this.ui.log.scrollHeight;
      if (!record) return;
      this.transcript.push(record);
      this.transcript = this.transcript.slice(-HISTORY_LIMIT);
      this.persist();
    }

    clearLog() {
      this.ui.log.replaceChildren();
      this.transcript = [];
      this.persist();
    }

    clearScreen(track = true) {
      const terminalStyle = getComputedStyle(this.ui.terminal);
      const verticalPadding =
        parseFloat(terminalStyle.paddingTop) + parseFloat(terminalStyle.paddingBottom);
      const bannerHeight = this.ui.terminal.querySelector('.banner')?.offsetHeight ?? 0;
      const viewportHeight = Math.max(
        1,
        this.ui.terminal.clientHeight - verticalPadding - bannerHeight - this.ui.form.offsetHeight,
      );
      const screen = makeElement('div', 'clear-screen');
      screen.style.height = `${viewportHeight}px`;
      this.append(screen, track ? { type: 'clear-screen' } : null);
    }

    persist() {
      try {
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({
            currentDirectory: this.currentDirectory,
            previousDirectory: this.previousDirectory,
            history: this.history.slice(-HISTORY_LIMIT),
            transcript: this.transcript.slice(-HISTORY_LIMIT),
          }),
        );
      } catch (error) {
        // Storage may be unavailable in private or restricted browser contexts.
      }
    }

    restore() {
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
        if (!saved) return;

        const savedDirectory = saved.currentDirectory === '' ? '/home' : saved.currentDirectory;
        const fallbackDirectory = this.entriesIn('/home') ? '/home' : '/';
        this.currentDirectory = this.entriesIn(savedDirectory) ? savedDirectory : fallbackDirectory;
        const previous = saved.previousDirectory === '' ? '/home' : saved.previousDirectory;
        this.previousDirectory =
          typeof previous === 'string' && this.entriesIn(previous) ? previous : null;
        this.history = Array.isArray(saved.history)
          ? saved.history.filter((item) => typeof item === 'string').slice(-HISTORY_LIMIT)
          : [];
        this.historyCursor = this.history.length;
        this.transcript = Array.isArray(saved.transcript)
          ? saved.transcript.slice(-HISTORY_LIMIT)
          : [];

        this.ui.prompt.textContent = this.promptText();
        if (!this.transcript.length) return;
        const warning = NOT_FOUND_PATH ? [...this.ui.log.childNodes] : [];
        this.ui.log.replaceChildren();
        this.transcript.forEach((record) => this.renderRecord(record));
        if (warning.length) {
          this.ui.log.append(...warning);
          this.ui.log.scrollTop = this.ui.log.scrollHeight;
        }
      } catch (error) {
        // Ignore malformed or inaccessible storage and start fresh.
      }
    }

    renderRecord(record) {
      if (record.type === 'line') {
        this.append(makeElement('p', `ln ${record.className ?? ''}`.trim(), record.text));
      } else if (record.type === 'link') {
        const line = makeElement('p', 'ln go');
        const link = makeElement('a', '', record.text);
        link.href = record.url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        line.append(link);
        this.append(line);
      } else if (record.type === 'echo') {
        const line = makeElement('p', 'ln');
        line.append(makeElement('span', 'pr', record.prompt), ` ${record.command}`);
        this.append(line);
      } else if (record.type === 'help') {
        this.commandSet.apps.help.show(false);
      } else if (record.type === 'clear-screen') {
        this.clearScreen(false);
      } else if (record.type === 'listing' && Array.isArray(record.entries)) {
        this.writeListing(record.entries, false);
      }
    }

    handleKey(event) {
      if (this.commandSet.apps.snake.handleKey(event)) return;

      if (event.key === ']' && !this.ui.input.value && this.commandSet.apps.chat.session) {
        event.preventDefault();
        this.commandSet.apps.chat.handleBracket();
        return;
      }

      if (event.ctrlKey && event.key.toLowerCase() === 'c') {
        event.preventDefault();
        this.echo(`${this.ui.input.value}^C`);
        this.autocomplete.clear();
        this.commandSet.apps.chat.cancel();
        return;
      }

      if (event.key === 'Enter' && this.autocomplete.accept()) {
        event.preventDefault();
        return;
      }

      const actions = {
        ArrowUp: () => this.autocomplete.recall(-1),
        ArrowDown: () => this.autocomplete.recall(1),
        Tab: () => this.autocomplete.complete(),
        Escape: () => this.autocomplete.clear(),
      };
      if (!actions[event.key]) return;
      event.preventDefault();
      actions[event.key]();
    }
  }

  if (NOT_FOUND_PATH) {
    const request = makeElement('p', 'ln');
    request.append(makeElement('span', 'pr', 'david:~$'), ` ${NOT_FOUND_PATH}`);
    byId('log').replaceChildren(
      request,
      makeElement('p', 'ln err', `zsh: no such file or directory: ${NOT_FOUND_PATH}`),
      makeElement('p', 'hint', '# that page did not exist. use help for more details.'),
    );
  } else {
    byId('log').replaceChildren(makeElement('p', 'hint', '# use help for more details.'));
  }

  fetch('/common/footer.html')
    .then((response) => response.text())
    .then((html) => {
      byId('site-footer').innerHTML = html;
    })
    .catch(() => {});

  const startTerminal = async () => {
    try {
      await window.KrasowTerminalFileSystem.loadManifest('/terminal/fs/manifest.json', {
        directories: DIRECTORIES,
        fileRoutes: FILE_ROUTES,
        pageSources: PAGE_SOURCES,
        textPaths: TEXT_PATHS,
      });
    } catch (error) {
      // Start with an empty virtual filesystem if the generated manifest cannot be loaded.
    }
    new Terminal().start();
  };

  startTerminal();
})();
