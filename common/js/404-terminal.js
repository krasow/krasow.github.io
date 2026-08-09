(() => {
  'use strict';

  const ROUTES = {
    home: '/index.html',
    projects: '/index.html#projects',
    publications: '/index.html#publications',
    posters: '/pages/presentations.html',
    experience: '/index.html#experience',
    cv: '/pages/cv.html',
    about: '/pages/about.html',
    news: '/pages/news.html',
    presentations: '/pages/presentations.html',
    resume: 'https://krasow.dev/assets/documents/Krasowska_David_resume.pdf',
    github: 'https://github.com/krasow',
    zoom: 'https://northwestern.zoom.us/my/krasow',
    contact: '/assets/documents/Krasowska_David_contact.vcf',
  };

  const FILES = {
    'ai-notice.md': '/assets/documents/notice/ai-notice.md',
    'contact.vcf': ROUTES.contact,
    'resume.pdf': ROUTES.resume,
  };
  const READABLE_FILES = new Set(['ai-notice.md', 'contact.vcf']);

  const FOLDERS = {
    projects: [
      ['cunumeric', '/pages/show.html?page=cunumeric'],
      ['legionpim', '/pages/show.html?page=legionpim'],
      ['compression', '/pages/show.html?page=compression'],
    ],
    publications: [
      ['VILLAGE25', 'https://www.mccormick.northwestern.edu/computer-science/documents/nu-cs-2025-33.pdf'],
      ['CLUSTER23', 'https://www.computer.org/csdl/proceedings-article/cluster/2023/079200a247/1SfUsploNQQ'],
      ['IJHPCA23', 'https://journals.sagepub.com/doi/abs/10.1177/10943420231179417'],
      ['DRBSD21', 'https://ieeexplore.ieee.org/abstract/document/9652575'],
    ],
    posters: [
      ['CSGF26', '/assets/documents/posters/2026_csgf_krasowska.pdf'],
      ['GCASR26', '/assets/documents/posters/2026_gcasr_krasowska.pdf'],
      ['CSGF25', '/assets/documents/posters/2025_csgf_krasoska.pdf'],
      ['SC22', '/assets/documents/posters/poster_krasowska.pdf'],
    ],
    presentations: [
      ['JULIACON25', '/assets/documents/slides/2025/juliacon.pdf'],
      ['LEGION24', '/assets/documents/slides/2024/legion24.pdf'],
      ['CONSTELLATION23', '/assets/documents/slides/2023/Constellation_Krasowska.pdf'],
      ['GRADSCHOOL22', '/assets/documents/slides/2022/grad_school_talk_dube_krasowska.pdf'],
      ['SC22', '/assets/documents/slides/2022/best_krasowska.pdf'],
      ['SASSY22', '/assets/documents/slides/2022/prediction_lossy_compression_krasowska.pdf'],
      ['DRBSD21', '/assets/documents/slides/2021/DRBSB-7-Krasowska.pdf'],
    ],
    scripts: [
      ['cunumeric-install.sh', '/scripts/cunumeric-install.sh'],
    ],
  };

  const ROOT_ENTRIES = [
    'about', 'ai-notice.md', 'contact.vcf', 'cv', 'experience', 'home', 'news',
    'posters/', 'presentations/', 'projects/', 'publications/',
    'resume.pdf', 'scripts/',
  ];

  const RESPONSES = {
    name: 'David Krasowska',
    email: 'david@krasow.dev',
    phone: '+1 (224) 355-2715',
    location: 'Chicago, IL',
    whoami: 'David Krasowska',
  };

  const HIDDEN_RESPONSES = { hi: 'Hello!', hello: 'Hello!' };
  const STORAGE_KEY = 'krasow-terminal-state';
  const HISTORY_LIMIT = 10;
  const PAGE_PATH = location.pathname || '/';
  const IS_TERMINAL_PAGE = PAGE_PATH.replace(/\/+$/, '') === '/terminal';

  const HELP = [
    ['Navigation', [
      ['ls [folder]', 'list pages or folder contents'],
      ['cd folder · cd .. · cd -', 'change directory'],
      ['pwd · tree', 'inspect the current directory'],
      ['cat file', 'read a text file'],
      ['show script', 'print an install command'],
    ]],
    ['Information', [
      ['whoami · name', 'show name'],
      ['email · phone · location', 'show contact details'],
    ]],
    ['Links', [
      ['github · zoom', 'open an external page'],
      ['resume · contact', 'open or download a document'],
    ]],
    ['Controls', [
      ['clear', 'clear the terminal'],
      ['Esc · Ctrl+C', 'cancel current input'],
      ['↑ / ↓ · Tab', 'history and autocomplete'],
    ]],
  ];

  const byId = (id) => document.getElementById(id);
  const makeElement = (tag, className, text = '') => {
    const node = document.createElement(tag);
    node.className = className;
    node.textContent = text;
    return node;
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

      this.currentDirectory = '';
      this.previousDirectory = null;
      this.history = [];
      this.historyCursor = 0;
      this.completionCycle = null;
      this.transcript = [];

      this.folderMaps = Object.fromEntries(
        Object.entries(FOLDERS).map(([name, entries]) => [name, new Map(entries)]),
      );
      this.entryRoutes = new Map(Object.values(FOLDERS).flat());
      this.completions = this.buildCompletions();
      this.commands = this.buildCommands();
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
      this.ui.input.focus();
    }

    buildCommands() {
      return new Map([
        ['clear', (args) => this.withArity(args, 0, () => this.clearLog())],
        ['help', (args) => this.withArity(args, 0, () => this.showHelp())],
        ['pwd', (args) => this.withArity(args, 0, () => this.write(this.path(), 'pth'))],
        ['tree', (args) => this.withArity(args, 0, () => this.showTree())],
        ['cat', (args) => this.withArity(args, 1, () => this.readFile(args[0]))],
        ['ls', (args) => {
          this.list(args.join(' '));
          return true;
        }],
        ['cd', (args) => this.withMaximumArity(args, 1, () => this.changeDirectory(args[0] ?? ''))],
        ['show', (args) => this.withArity(args, 1, () => this.showScript(args[0]))],
      ]);
    }

    buildCompletions() {
      const folderNames = Object.keys(FOLDERS);
      const entries = Object.entries(FOLDERS).flatMap(([folder, items]) => (
        items.flatMap(([name]) => [name, `${folder}/${name}`])
      ));
      return [...new Set([
        ...['help', 'clear', 'pwd', 'tree', 'whoami', 'cat', 'show', 'ls', 'cd', 'cd ..', 'cd -'],
        'cat ai-notice.md',
        'cat contact.vcf',
        ...folderNames.flatMap((folder) => [`ls ${folder}`, `cd ${folder}`]),
        ...FOLDERS.scripts.map(([name]) => `show ${name}`),
        ...Object.keys(ROUTES),
        ...Object.keys(FILES),
        ...Object.keys(RESPONSES),
        ...entries,
      ])];
    }

    withArity(args, count, action) {
      if (args.length !== count) return false;
      action();
      return true;
    }

    withMaximumArity(args, count, action) {
      if (args.length > count) return false;
      action();
      return true;
    }

    execute(raw) {
      const command = raw.trim();
      this.echo(command);
      if (!command) return;

      this.history.push(command);
      this.history = this.history.slice(-HISTORY_LIMIT);
      this.historyCursor = this.history.length;
      this.persist();

      const [name, ...args] = command.split(/\s+/);
      if (this.commands.get(name)?.(args)) return;

      const response = RESPONSES[name] ?? HIDDEN_RESPONSES[name];
      if (response && !args.length) {
        this.write(response, 'pth');
        return;
      }

      const url = this.resolve(command);
      if (url) this.navigate(url);
      else this.write(`zsh: no such command, file, or directory: ${command}`, 'err');
    }

    path() {
      return this.currentDirectory ? `~/${this.currentDirectory}` : '~';
    }

    promptText() {
      return `david:${this.path()}$`;
    }

    directoryFromPath(path) {
      const clean = path.trim().replace(/\/+$/g, '');
      if (!clean || clean === '~' || clean === '/') return '';
      if (clean === '.') return this.currentDirectory;
      if (clean === '..') return '';
      return clean.replace(/^~?\//, '');
    }

    resolve(command) {
      const path = command.trim().replace(/^~?\/+|\/+$/g, '');
      const [folder, entry, extra] = path.split('/');
      if (entry && !extra) return this.folderMaps[folder]?.get(entry);
      if (entry) return undefined;
      return ROUTES[folder]
        ?? FILES[folder]
        ?? this.folderMaps[this.currentDirectory]?.get(folder)
        ?? this.entryRoutes.get(folder);
    }

    list(path) {
      const directory = path ? this.directoryFromPath(path) : this.currentDirectory;
      if (!directory) {
        this.write(ROOT_ENTRIES.join('   '), 'pth');
        return;
      }
      if (!FOLDERS[directory]) {
        this.write(`ls: ${path}: not a directory`, 'err');
        return;
      }

      const names = FOLDERS[directory].map(([name]) => name);
      if (directory === 'projects') names.sort((a, b) => a.localeCompare(b));
      this.write(names.join('   '), 'pth');
    }

    changeDirectory(target) {
      const requested = this.directoryFromPath(target);
      let next = requested;

      if (requested === '-') {
        if (this.previousDirectory === null) {
          this.write('cd: no previous directory', 'err');
          return;
        }
        next = this.previousDirectory;
      } else if (requested && !FOLDERS[requested]) {
        this.write(`cd: no such file or directory: ${target}`, 'err');
        return;
      }

      this.previousDirectory = this.currentDirectory;
      this.currentDirectory = next;
      this.ui.prompt.textContent = this.promptText();
      this.persist();
    }

    showScript(path) {
      const name = path.replace(/^scripts\//, '');
      const url = this.folderMaps.scripts.get(name);
      if (url) this.write(`curl -fsSL https://krasow.dev${url} | bash`, 'pth');
      else this.write(`show: no such script: ${path}`, 'err');
    }

    async readFile(path) {
      if (!READABLE_FILES.has(path)) {
        this.write(`cat: ${path}: no such text file`, 'err');
        return;
      }
      try {
        const response = await fetch(FILES[path]);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        this.write((await response.text()).trim(), 'pth');
      } catch (error) {
        this.write(`cat: ${path}: unable to read file`, 'err');
      }
    }

    showTree() {
      const lines = ['.'];
      if (this.currentDirectory) {
        this.appendTreeEntries(lines, FOLDERS[this.currentDirectory]);
      } else {
        ROOT_ENTRIES.forEach((entry, index) => {
          const isLast = index === ROOT_ENTRIES.length - 1;
          lines.push(`${isLast ? '└──' : '├──'} ${entry}`);
          const folder = entry.endsWith('/') ? entry.slice(0, -1) : '';
          if (folder) this.appendTreeEntries(lines, FOLDERS[folder], isLast ? '    ' : '│   ');
        });
      }
      this.write(lines.join('\n'), 'pth');
    }

    appendTreeEntries(lines, entries, prefix = '') {
      entries.forEach(([name], index) => {
        const branch = index === entries.length - 1 ? '└──' : '├──';
        lines.push(`${prefix}${branch} ${name}`);
      });
    }

    showHelp(track = true) {
      const help = makeElement('div', 'ln help');
      HELP.forEach(([heading, rows]) => {
        help.append(makeElement('span', 'help-section', heading));
        rows.forEach(([command, description]) => help.append(
          makeElement('span', 'help-command', command),
          makeElement('span', 'help-description', description),
        ));
      });
      help.append(makeElement('span', 'help-note', 'Type any item shown by ls to open it.'));
      this.append(help, track ? { type: 'help' } : null);
    }

    navigate(url) {
      this.writeLink(
        url,
        url.endsWith('.vcf') ? '→ downloading contact.vcf' : `→ ${url}`,
      );
      setTimeout(() => { location.href = url; }, 120);
    }

    write(text, className = '') {
      this.append(
        makeElement('p', `ln ${className}`.trim(), text),
        { type: 'line', text, className },
      );
    }

    writeLink(url, text) {
      const line = makeElement('p', 'ln go');
      const link = makeElement('a', '', text);
      link.href = url;
      line.append(link);
      this.append(line, { type: 'link', url, text });
    }

    echo(command) {
      const line = makeElement('p', 'ln');
      const prompt = this.promptText();
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

    persist() {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({
          currentDirectory: this.currentDirectory,
          previousDirectory: this.previousDirectory,
          history: this.history.slice(-HISTORY_LIMIT),
          transcript: this.transcript.slice(-HISTORY_LIMIT),
        }));
      } catch (error) {
        // Storage may be unavailable in private or restricted browser contexts.
      }
    }

    restore() {
      try {
        const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
        if (!saved) return;

        this.currentDirectory = FOLDERS[saved.currentDirectory]
          ? saved.currentDirectory
          : '';
        this.previousDirectory = saved.previousDirectory === ''
          || FOLDERS[saved.previousDirectory]
          ? saved.previousDirectory
          : null;
        this.history = Array.isArray(saved.history)
          ? saved.history.filter((item) => typeof item === 'string').slice(-HISTORY_LIMIT)
          : [];
        this.historyCursor = this.history.length;
        this.transcript = Array.isArray(saved.transcript)
          ? saved.transcript.slice(-HISTORY_LIMIT)
          : [];

        this.ui.prompt.textContent = this.promptText();
        if (!this.transcript.length) return;
        const warning = IS_TERMINAL_PAGE ? [] : [...this.ui.log.childNodes];
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
        line.append(link);
        this.append(line);
      } else if (record.type === 'echo') {
        const line = makeElement('p', 'ln');
        line.append(makeElement('span', 'pr', record.prompt), ` ${record.command}`);
        this.append(line);
      } else if (record.type === 'help') {
        this.showHelp(false);
      }
    }

    handleKey(event) {
      if (event.ctrlKey && event.key.toLowerCase() === 'c') {
        event.preventDefault();
        this.echo(`${this.ui.input.value}^C`);
        this.clearInput();
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
      this.historyCursor = Math.max(
        0,
        Math.min(this.history.length, this.historyCursor + delta),
      );
      this.ui.input.value = this.history[this.historyCursor] ?? '';
      this.hideCompletions();
    }

    complete() {
      const typed = this.ui.input.value.trim();
      if (this.completionCycle?.value === typed) {
        this.cycleCompletion();
        return;
      }

      const matches = (typed
        ? this.completions.filter((command) => command.startsWith(typed))
        : []).sort((a, b) => a.localeCompare(b));

      if (!matches.length) {
        this.hideCompletions();
      } else if (matches.length === 1) {
        this.ui.input.value = matches[0];
        this.hideCompletions();
      } else {
        this.completeMultiple(typed, matches);
      }
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

      const directoryCommand = /^(ls|cd)( |$)/.test(typed);
      const choices = directoryCommand
        ? matches.filter((match) => match.includes(' ')).map((match) => match.split(' ')[1])
        : [...matches];
      const active = this.completionCycle.index < 0
        ? ''
        : this.completionCycle.matches[this.completionCycle.index];
      this.showCompletions(choices, directoryCommand ? active.split(' ')[1] : active);
    }

    cycleCompletion() {
      const cycle = this.completionCycle;
      cycle.index = (cycle.index + 1) % cycle.matches.length;
      cycle.value = cycle.matches[cycle.index];
      this.ui.input.value = cycle.value;

      const directoryCommand = /^(ls|cd)( |$)/.test(cycle.matches[0]);
      const choices = directoryCommand
        ? cycle.matches.filter((match) => match.includes(' ')).map((match) => match.split(' ')[1])
        : cycle.matches;
      const active = directoryCommand ? cycle.value.split(' ')[1] : cycle.value;
      this.showCompletions(choices, active);
    }

    showCompletions(choices, active = '') {
      this.ui.autocomplete.replaceChildren(...choices.map((choice) => (
        makeElement(
          'span',
          `autocomplete-choice${choice === active ? ' active' : ''}`,
          choice,
        )
      )));
      this.ui.autocomplete.hidden = !choices.length;
    }

    hideCompletions() {
      this.completionCycle = null;
      this.ui.autocomplete.hidden = true;
      this.ui.autocomplete.replaceChildren();
    }
  }

  if (IS_TERMINAL_PAGE) {
    document.title = 'Terminal | David Krasowska';
    const banner = document.querySelector('.glitch');
    banner.textContent = 'krasow.dev';
    banner.dataset.t = 'krasow.dev';
    banner.setAttribute('aria-label', 'krasow.dev');
    banner.classList.add('plain');
    byId('log').replaceChildren(
      makeElement('p', 'hint', '# use help for more details.'),
    );
  } else {
    document.querySelectorAll('.req').forEach((element) => { element.textContent = PAGE_PATH; });
  }

  fetch('/common/footer.html')
    .then((response) => response.text())
    .then((html) => { byId('site-footer').innerHTML = html; })
    .catch(() => {});

  new Terminal().start();
})();
