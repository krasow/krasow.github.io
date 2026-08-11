(() => {
  'use strict';

  const SHORTCUTS = {
    github: 'https://github.com/krasow',
    zoom: 'https://northwestern.zoom.us/my/krasow',
  };
  const PAGE_SOURCES = {};
  const READABLE_FILES = {};

  const HIDDEN_FILES = {};
  const DIRECTORIES = {
    '/': [],
  };
  const FILE_ROUTES = new Map();
  const TEXT_PATHS = new Set([
    ...Object.keys(READABLE_FILES).map((name) => `/home/${name}`),
    ...Object.keys(PAGE_SOURCES).map((name) => `/home/${name}`),
    ...Object.keys(HIDDEN_FILES),
  ]);

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
  const globRegex = (pattern) => {
    const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&');
    return new RegExp(`^${escaped.replace(/\*/g, '.*').replace(/\?/g, '.')}$`);
  };

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
        hiddenFiles: HIDDEN_FILES,
        storageKey: REMOVED_PATHS_KEY,
      });

      this.currentDirectory = '/home';
      this.previousDirectory = null;
      this.history = [];
      this.historyCursor = 0;
      this.completionCycle = null;
      this.transcript = [];
      this.commandSet = new window.KrasowTerminalCommands.TerminalCommands(this, {
        shortcuts: SHORTCUTS,
        fileRoutes: FILE_ROUTES,
        hiddenFiles: HIDDEN_FILES,
        directories: DIRECTORIES,
      });

      this.completions = this.commandSet.completions();
      this.commands = this.commandSet.commands();
      this.restore();
    }

    start() {
      this.ui.form.addEventListener('submit', (event) => {
        event.preventDefault();
        this.hideCompletions();
        this.execute(this.ui.input.value);
        this.ui.input.value = '';
      });
      this.ui.input.addEventListener('input', () => this.hideCompletions());
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

    resolvePath(input) {
      let path = input.trim();
      if (!path || path === '~') return '/home';
      if (path.startsWith('~/')) path = `/home/${path.slice(2)}`;
      else if (!path.startsWith('/')) path = `${this.currentDirectory}/${path}`;

      const parts = [];
      path.split('/').forEach((part) => {
        if (!part || part === '.') return;
        if (part === '..') parts.pop();
        else parts.push(part);
      });
      return `/${parts.join('/')}`;
    }

    resolve(command) {
      const path = this.resolvePath(command);
      return SHORTCUTS[command] ?? (this.trash.contains(path) ? null : FILE_ROUTES.get(path));
    }

    writeListing(entries, track = true) {
      const listing = makeElement('div', 'ln pth ls-grid');
      listing.append(...entries.map((name) => makeElement('span', 'ls-entry', displayFile(name))));
      this.append(listing, track ? { type: 'listing', entries } : null);
    }

    displayFile(name) {
      return displayFile(name);
    }

    matchingEntries(path) {
      const separator = path.lastIndexOf('/');
      const pattern = path.slice(separator + 1);
      const directoryPath = separator < 0 ? '.' : path.slice(0, separator) || '/';
      const directory = this.resolvePath(directoryPath);
      const entries = this.entriesIn(directory);
      const prefix = separator < 0 ? '' : `${directory === '/' ? '' : directory}/`;

      if (!entries) return { directoryPath, prefix, matches: null };

      const matches = entries.filter((entry) => globRegex(pattern).test(entry));
      return { directoryPath, prefix, matches };
    }

    expandPath(path) {
      if (!/[*?]/.test(path)) return [path];
      const { prefix, matches } = this.matchingEntries(path);
      return (matches ?? []).map((name) => `${prefix}${name}`);
    }

    entriesIn(directory) {
      if (this.trash.contains(directory) || !DIRECTORIES[directory]) return null;
      return DIRECTORIES[directory].filter((name) => {
        if (name.startsWith('.')) return false;
        const child = `${directory === '/' ? '' : directory}/${name.replace(/\/$/, '')}`;
        return !this.trash.contains(child);
      });
    }

    async fileText(path) {
      const resolved = this.resolvePath(path);
      if (this.trash.contains(resolved)) throw new Error('ENOENT');
      if (DIRECTORIES[resolved]) throw new Error('EISDIR');
      const source = this.textSource(path);
      if (!source) throw new Error('ENOENT');
      return this.loadText(source);
    }

    textSource(path) {
      const absolutePath = this.resolvePath(path);
      if (this.trash.contains(absolutePath)) return null;
      if (HIDDEN_FILES[absolutePath]) return { url: HIDDEN_FILES[absolutePath] };
      const homePath = absolutePath.replace(/^\/home\//, '');
      if (PAGE_SOURCES[homePath]) return PAGE_SOURCES[homePath];

      const url =
        READABLE_FILES[homePath] ??
        (TEXT_PATHS.has(absolutePath) ? FILE_ROUTES.get(absolutePath) : null);
      return url ? { url } : null;
    }

    async loadText(source) {
      const response = await fetch(source.url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const content = (await response.text()).trim();
      if (!source.selector) return content;

      const document = new DOMParser().parseFromString(content, 'text/html');
      const text = [...document.querySelectorAll(source.selector)]
        .map((section) => this.pageText(section))
        .filter(Boolean)
        .join('\n\n')
        .replace(/\n{3,}/g, '\n\n');
      if (!text) throw new Error('No readable content');
      return text;
    }

    pageText(section) {
      const copy = section.cloneNode(true);
      copy.querySelectorAll('.cv-date span + span').forEach((span) => span.before(' – '));
      copy
        .querySelectorAll('.pub-date br')
        .forEach((breakElement) => breakElement.replaceWith(' – '));
      copy
        .querySelectorAll(
          ['br', 'div', 'p', 'li', 'hr', 'h1', 'h2', 'h3', '.cv-title', '.cv-org', '.cv-desc'].join(
            ',',
          ),
        )
        .forEach((element) => element.after('\n'));

      return copy.textContent
        .replace(/[ \t]+/g, ' ')
        .replace(/ *\n */g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
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
        this.clearInput();
        this.commandSet.apps.chat.cancel();
        return;
      }

      const [command, target] = this.ui.input.value.trim().split(/\s+/);
      if (
        event.key === 'Enter' &&
        command === 'open' &&
        !this.ui.autocomplete.hidden &&
        this.entriesIn(this.resolvePath(target))
      ) {
        event.preventDefault();
        this.completionCycle = null;
        this.complete();
        return;
      }

      const actions = {
        ArrowUp: () => this.recall(-1),
        ArrowDown: () => this.recall(1),
        Tab: () => this.complete(),
        Escape: () => this.clearInput(),
      };
      if (!actions[event.key]) return;
      event.preventDefault();
      actions[event.key]();
    }

    clearInput() {
      this.ui.input.value = '';
      this.hideCompletions();
    }

    recall(delta) {
      this.historyCursor = Math.max(0, Math.min(this.history.length, this.historyCursor + delta));
      this.ui.input.value = this.history[this.historyCursor] ?? '';
      this.hideCompletions();
    }

    complete() {
      const typed = this.ui.input.value.trimStart();
      if (!typed.trim()) {
        this.hideCompletions();
        return;
      }
      if (this.completionCycle?.value === typed) {
        this.cycleCompletion();
        return;
      }

      const tokenStart = typed.lastIndexOf(' ') + 1;
      const path = typed.slice(tokenStart);
      const pathCommand = /^(cat|cd|copy|download|find|grep|ls|open|rm|show|wc)\b/.test(typed)
        ? typed.split(/\s/)[0]
        : !typed.includes(' ')
          ? ''
          : null;
      const paths =
        pathCommand !== null
          ? this.pathCompletions(path, pathCommand).map(
              (candidate) => `${typed.slice(0, tokenStart)}${candidate}`,
            )
          : [];
      const matches = [...new Set([...this.completions, ...paths])]
        .filter((command) => command.startsWith(typed))
        .sort((a, b) => a.localeCompare(b));

      if (!matches.length) {
        this.hideCompletions();
      } else if (matches.length === 1) {
        this.ui.input.value = matches[0];
        this.hideCompletions();
      } else {
        this.completeMultiple(typed, matches);
      }
    }

    pathCompletions(path, command) {
      const separator = path.lastIndexOf('/');
      let candidates;
      if (separator < 0) {
        candidates = this.completionEntries(this.currentDirectory, path).map((name) => ({
          name,
          path: this.resolvePath(name),
        }));
      } else {
        const directory = this.resolvePath(path.slice(0, separator) || '/');
        const prefix = path.slice(0, separator + 1);
        candidates = this.completionEntries(directory, path.slice(separator + 1)).map((name) => ({
          name: `${prefix}${name}`,
          path: this.resolvePath(`${prefix}${name}`),
        }));
      }

      return candidates
        .filter(({ name }) => name.startsWith(path))
        .filter(({ path: candidate }) => {
          if (command === 'cd') return Boolean(DIRECTORIES[candidate]);
          if (command === 'open' || command === 'download') {
            const openable = [...FILE_ROUTES.keys(), ...Object.keys(HIDDEN_FILES)];
            const matchesPath =
              openable.includes(candidate) ||
              ((path.includes('/') || path.startsWith('.')) &&
                openable.some((target) => target.startsWith(`${candidate}/`)));
            if (!matchesPath) return false;
            if (command === 'open' && /\.vcf$/i.test(candidate)) return false;
            if (command === 'download' && !DIRECTORIES[candidate]) {
              return /\.(pdf|sh|vcf)$/i.test(candidate);
            }
            return true;
          }
          if (!['cat', 'copy', 'grep', 'wc'].includes(command)) return true;
          return (
            TEXT_PATHS.has(candidate) ||
            ((path.includes('/') || path.startsWith('.')) &&
              [...TEXT_PATHS].some((textPath) => textPath.startsWith(`${candidate}/`)))
          );
        })
        .map(({ name }) => name);
    }

    completionEntries(directory, partial) {
      const visible = this.entriesIn(directory) ?? [];
      if (!partial.startsWith('.')) return visible;

      const prefix = directory === '/' ? '/' : `${directory}/`;
      const hidden = Object.keys(DIRECTORIES)
        .filter((path) => path.startsWith(`${prefix}.`))
        .filter((path) => !this.trash.contains(path))
        .map((path) => path.slice(prefix.length).split('/')[0])
        .filter(Boolean)
        .map((name) => `${name}/`);
      return [...new Set([...visible, ...hidden])];
    }

    completeMultiple(typed, matches) {
      const prefix = matches.reduce((common, match) => {
        let end = 0;
        while (end < common.length && common[end] === match[end]) end += 1;
        return common.slice(0, end);
      });

      if (prefix.length > typed.length) {
        this.ui.input.value = prefix;
        const index = matches.indexOf(prefix);
        this.completionCycle = { matches, index, value: prefix };
      } else {
        this.ui.input.value = matches[0];
        this.completionCycle = { matches, index: 0, value: matches[0] };
      }

      const active =
        this.completionCycle.index < 0
          ? ''
          : this.completionCycle.matches[this.completionCycle.index];
      this.showCompletionMenu(matches, active);
    }

    cycleCompletion() {
      const cycle = this.completionCycle;
      cycle.index = (cycle.index + 1) % cycle.matches.length;
      cycle.value = cycle.matches[cycle.index];
      this.ui.input.value = cycle.value;

      this.showCompletionMenu(cycle.matches, cycle.value);
    }

    showCompletionMenu(matches, active) {
      const commandPrefix = `${matches[0].split(' ')[0]} `;
      const slash = matches[0].lastIndexOf('/');
      const pathPrefix = slash < 0 ? '' : matches[0].slice(0, slash + 1);
      const prefix =
        pathPrefix && matches.every((match) => match.startsWith(pathPrefix))
          ? pathPrefix
          : commandPrefix;
      const label = (match) => (match.startsWith(prefix) ? match.slice(prefix.length) : match);
      this.showCompletions(matches.map(label), label(active));
    }

    showCompletions(choices, active = '') {
      this.ui.autocomplete.replaceChildren(
        ...choices.map((choice) =>
          makeElement('span', `autocomplete-choice${choice === active ? ' active' : ''}`, choice),
        ),
      );
      this.ui.autocomplete.hidden = !choices.length;
    }

    hideCompletions() {
      this.completionCycle = null;
      this.ui.autocomplete.hidden = true;
      this.ui.autocomplete.replaceChildren();
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
