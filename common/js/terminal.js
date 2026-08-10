(() => {
  'use strict';

  const ROUTES = {
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
    'about.pg': ROUTES.about,
    'ai-notice.md': '/assets/documents/notice/ai-notice.md',
    'contact.md': '/assets/documents/terminal/contact.md',
    'contact.vcf': ROUTES.contact,
    'cv.pg': ROUTES.cv,
    'education.md': '/assets/documents/terminal/education.md',
    'experience.pg': ROUTES.experience,
    'news.pg': ROUTES.news,
    'resume.pdf': ROUTES.resume,
    'summary.md': '/assets/documents/terminal/summary.md',
  };
  const SHORTCUTS = {
    github: ROUTES.github,
    zoom: ROUTES.zoom,
  };
  const PAGE_SOURCES = {
    'about.pg': { url: FILES['about.pg'], selector: '.holder' },
    'cv.pg': { url: FILES['cv.pg'], selector: '.holder' },
    'experience.pg': { url: '/homepage/experience.html', selector: '.holder' },
    'news.pg': { url: FILES['news.pg'], selector: '.holder' },
  };
  const READABLE_FILES = {
    'ai-notice.md': FILES['ai-notice.md'],
    'contact.md': FILES['contact.md'],
    'summary.md': FILES['summary.md'],
    'education.md': FILES['education.md'],
  };

  const FOLDERS = {
    projects: [
      ['compression', '/pages/show.html?page=compression'],
      ['cunumeric', '/pages/show.html?page=cunumeric'],
      ['legionpim', '/pages/show.html?page=legionpim'],
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
  const HIDDEN_FILES = {
    '/etc/hostname': '/assets/terminal/etc/hostname',
    '/etc/motd': '/assets/terminal/etc/motd',
    '/etc/os-release': '/assets/terminal/etc/os-release',
    '/home/.ssh/id_ed25519_krasow.pub': '/assets/terminal/home/.ssh/id_ed25519_krasow.pub',
    '/proc/version': '/assets/terminal/proc/version',
  };
  const HIDDEN_DIRECTORIES = {
    '/etc': ['hostname', 'motd', 'os-release'],
    '/home/.ssh': ['id_ed25519_krasow.pub'],
    '/proc': ['version'],
  };
  const ROOT_ENTRIES = [
    'about.pg', 'ai-notice.md', 'contact.md', 'contact.vcf', 'cv.pg', 'education.md',
    'experience.pg', 'news.pg',
    'posters/', 'presentations/', 'projects/', 'publications/',
    'resume.pdf', 'scripts/', 'summary.md',
  ];
  const ROOT_DIRECTORIES = ['etc/', 'home/', 'proc/'];
  const DIRECTORIES = {
    '/': ROOT_DIRECTORIES,
    '/home': ROOT_ENTRIES,
    ...Object.fromEntries(Object.entries(FOLDERS).map(([name, entries]) => [
      `/home/${name}`,
      entries.map(([entry]) => entry),
    ])),
    ...HIDDEN_DIRECTORIES,
  };
  const FILE_ROUTES = new Map([
    ...Object.entries(FILES).map(([name, url]) => [`/home/${name}`, url]),
    ...Object.entries(FOLDERS).flatMap(([folder, entries]) => (
      entries.filter(([, url]) => url).map(([name, url]) => [`/home/${folder}/${name}`, url])
    )),
  ]);
  const TEXT_PATHS = new Set([
    ...Object.keys(READABLE_FILES).map((name) => `/home/${name}`),
    ...Object.keys(PAGE_SOURCES).map((name) => `/home/${name}`),
    ...Object.keys(HIDDEN_FILES),
    ...FOLDERS.scripts.map(([name]) => `/home/scripts/${name}`),
  ]);

  const RESPONSES = {
    whoami: 'David Krasowska',
  };

  const HIDDEN_RESPONSES = { hi: 'Hello!', hello: 'Hello!' };
  const COMMAND_USAGE = {
    chat: 'chat [question]',
    clear: 'clear',
    help: 'help',
    pwd: 'pwd',
    tree: 'tree',
    theme: 'theme [light|dark]',
    cat: 'cat <file|pattern> [...]',
    ls: 'ls [folder|pattern]',
    cd: 'cd [folder|..|-]',
    show: 'show <script>',
    echo: 'echo [text]',
    grep: 'grep <pattern> <file|pattern> [...]',
    copy: 'copy <file>',
    wc: 'wc <file|pattern> [...]',
    open: 'open <file|page>',
    find: 'find [folder] [pattern]',
  };
  const STORAGE_KEY = 'krasow-terminal-state';
  const HISTORY_LIMIT = 10;
  const PAGE_PATH = location.pathname || '/';
  const IS_TERMINAL_PAGE = PAGE_PATH.replace(/\/+$/, '') === '/terminal';

  const HELP = [
    ['Navigation', [
      ['ls [folder|pattern]', 'list pages or matching files'],
      ['cd folder · cd .. · cd -', 'change directory'],
      ['pwd · tree', 'inspect the current directory'],
      ['find [folder] [pattern]', 'find files recursively'],
      ['open file|page', 'open a page or document'],
      ['cat file|pattern', 'read one or more text files'],
      ['grep pattern file', 'search one or more text files'],
      ['copy file', 'copy a text file'],
      ['wc file|pattern', 'count lines, words, and characters'],
      ['echo text', 'print text'],
      ['show script', 'print an install command'],
    ]],
    ['Information', [
      ['chat [question]', 'ask about David or enter chat mode'],
      ['whoami', 'show name'],
      ['cat contact.md', 'show contact details'],
    ]],
    ['Links', [
      ['github · zoom', 'open an external page'],
      ['resume.pdf · contact.vcf', 'open or download a document'],
    ]],
    ['Controls', [
      ['clear', 'clear the terminal'],
      ['theme [light|dark]', 'change color theme'],
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
  const fileIcon = (name) => {
    if (name.endsWith('/')) return '📁';
    if (name.endsWith('.pg')) return '🌐';
    if (name.endsWith('.md')) return '📝';
    if (name.endsWith('.pdf')) return '📕';
    if (name.endsWith('.sh')) return '⚙';
    if (name.endsWith('.pub')) return '🔑';
    if (name.endsWith('.vcf')) return '👤';
    return '📄';
  };
  const displayFile = (name) => `${fileIcon(name)}\u00a0${name}`;
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

      this.currentDirectory = '/home';
      this.previousDirectory = null;
      this.history = [];
      this.historyCursor = 0;
      this.completionCycle = null;
      this.transcript = [];
      this.chatModel = new window.KrasowChat.LocalChat();
      this.chatSession = false;
      this.chatMode = false;

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
        ['chat', (args) => {
          if (args.length) this.askChat(args.join(' '));
          else this.enterChat();
          return true;
        }],
        ['clear', (args) => this.withArity(args, 0, () => this.clearLog())],
        ['help', (args) => this.withArity(args, 0, () => this.showHelp())],
        ['pwd', (args) => this.withArity(args, 0, () => this.write(this.path(), 'pth'))],
        ['tree', (args) => this.withArity(args, 0, () => this.showTree())],
        ['theme', (args) => this.withMaximumArity(args, 1, () => this.setTheme(args[0]))],
        ['cat', (args) => {
          if (!args.length) return false;
          this.readFiles(args);
          return true;
        }],
        ['grep', (args) => {
          if (args.length < 2) return false;
          this.grep(args[0], args.slice(1));
          return true;
        }],
        ['copy', (args) => this.withArity(args, 1, () => this.copyFile(args[0]))],
        ['wc', (args) => {
          if (!args.length) return false;
          this.countFiles(args);
          return true;
        }],
        ['ls', (args) => {
          this.list(args.join(' '));
          return true;
        }],
        ['cd', (args) => this.withMaximumArity(args, 1, () => this.changeDirectory(args[0] ?? ''))],
        ['open', (args) => this.withArity(args, 1, () => this.openPath(args[0]))],
        ['find', (args) => this.withMaximumArity(args, 2, () => this.find(args))],
        ['show', (args) => this.withArity(args, 1, () => this.showScript(args[0]))],
        ['echo', (args) => {
          this.write(args.join(' '), 'pth');
          return true;
        }],
      ]);
    }

    buildCompletions() {
      return [...new Set([
        ...['help', 'chat', 'clear', 'pwd', 'tree', 'whoami', 'cat', 'grep', 'copy', 'wc', 'open', 'find', 'show', 'echo', 'ls', 'cd', 'cd ..', 'cd -'],
        'theme',
        'theme light',
        'theme dark',
        ...FOLDERS.scripts.map(([name]) => `show ${name}`),
        ...Object.keys(SHORTCUTS),
        ...Object.keys(RESPONSES),
        ...ROOT_ENTRIES,
      ])];
    }

    async askChat(question) {
      const thinking = makeElement('p', 'ln hint', 'local model: thinking…');
      this.append(thinking);

      try {
        const result = await this.chatModel.ask(question);
        thinking.remove();
        this.write(result.answer, 'pth');
        if (result.hint) this.write(result.hint, 'hint');
        result.links?.forEach(({ label, url }) => this.writeLink(url, `→ ${label}: ${url}`));
        if (result.command) this.runChatCommand(result.command);
      } catch (error) {
        thinking.remove();
        this.write('chat: the local knowledge model could not be loaded', 'err');
      }
    }

    enterChat() {
      this.chatSession = true;
      this.chatMode = true;
      this.ui.prompt.textContent = this.promptText();
      this.write('Ask me about David. Type `exit` to return to the terminal.', 'hint');
    }

    leaveChat() {
      this.chatSession = false;
      this.chatMode = false;
      this.ui.prompt.textContent = this.promptText();
      this.write('leaving chat', 'hint');
    }

    toggleChatShell() {
      this.chatMode = !this.chatMode;
      this.ui.prompt.textContent = this.promptText();
    }

    showChatHelp() {
      this.write([
        'Ask about David using natural language. For example:',
        '  what does he work on?',
        '  where did he study?',
        '  who is his advisor?',
        '  what publications does he have?',
        '  what is his work on PIM?',
        '  how can I contact him?',
        '',
        'Commands: help · ] toggle shell mode · exit · quit · Ctrl+C',
      ].join('\n'), 'pth');
    }

    runChatCommand(command) {
      this.echo(command, 'david:~$');
      const [name, ...args] = command.split(/\s+/);
      this.commands.get(name)?.(args);
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

      if (this.chatSession && !this.chatMode && command === ']') {
        this.toggleChatShell();
        return;
      }

      if (this.chatMode) {
        const chatCommand = command.toLowerCase();
        if (['exit', 'quit'].includes(chatCommand)) this.leaveChat();
        else if (['help', '?'].includes(chatCommand)) this.showChatHelp();
        else if (['hello', 'hi', 'hey'].includes(chatCommand)) {
          this.write('Hello! Ask me anything about David, or type `help` for examples.', 'pth');
        }
        else if (command === ']') this.toggleChatShell();
        else this.askChat(command);
        return;
      }

      const [name, ...args] = command.split(/\s+/);
      const handler = this.commands.get(name);
      if (handler) {
        if (handler(args)) return;
        this.write(`${name}: usage: ${COMMAND_USAGE[name]}`, 'err');
        return;
      }

      const response = RESPONSES[name] ?? HIDDEN_RESPONSES[name];
      if (response && !args.length) {
        this.write(response, 'pth');
        return;
      }

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
      return this.chatMode ? 'david:chat>' : `david:${this.path()}$`;
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
      return SHORTCUTS[command] ?? FILE_ROUTES.get(this.resolvePath(command));
    }

    list(path) {
      if (/[*?]/.test(path)) {
        this.listMatches(path);
        return;
      }
      const directory = path ? this.resolvePath(path) : this.currentDirectory;
      const entries = this.entriesIn(directory);
      if (!entries) {
        this.write(`ls: ${path}: not a directory`, 'err');
        return;
      }
      this.write(entries.map(displayFile).join('   '), 'pth');
    }

    listMatches(path) {
      const { directoryPath, matches } = this.matchingEntries(path);
      if (matches === null) {
        this.write(`ls: ${directoryPath}: not a directory`, 'err');
      } else if (matches.length) {
        this.write(matches.map(displayFile).join('   '), 'pth');
      } else {
        this.write(`ls: no matches found: ${path}`, 'err');
      }
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
      return DIRECTORIES[directory] ? [...DIRECTORIES[directory]] : null;
    }

    changeDirectory(target) {
      const requested = target === '-' ? '-' : this.resolvePath(target);
      let next = requested;

      if (requested === '-') {
        if (this.previousDirectory === null) {
          this.write('cd: no previous directory', 'err');
          return;
        }
        next = this.previousDirectory;
      } else if (!this.entriesIn(requested)) {
        this.write(`cd: no such file or directory: ${target}`, 'err');
        return;
      }

      this.previousDirectory = this.currentDirectory;
      this.currentDirectory = next;
      this.ui.prompt.textContent = this.promptText();
      this.persist();
    }

    openPath(target) {
      const path = this.resolvePath(target);
      if (this.entriesIn(path)) {
        this.write(`open: ${target}: is a directory`, 'err');
        return;
      }
      const url = SHORTCUTS[target] ?? FILE_ROUTES.get(path) ?? HIDDEN_FILES[path];
      if (url) this.navigate(url);
      else this.write(`open: ${target}: no such file or page`, 'err');
    }

    find(args) {
      const hasDirectory = args.length === 2
        || (args[0] && this.entriesIn(this.resolvePath(args[0])));
      const directory = hasDirectory ? this.resolvePath(args[0]) : this.currentDirectory;
      const pattern = args[hasDirectory ? 1 : 0] ?? '*';
      if (!this.entriesIn(directory)) {
        this.write(`find: ${args[0]}: not a directory`, 'err');
        return;
      }

      const walk = (folder) => this.entriesIn(folder).flatMap((name) => {
        const path = `${folder === '/' ? '' : folder}/${name.replace(/\/$/, '')}`;
        return [path, ...(name.endsWith('/') ? walk(path) : [])];
      });
      const matches = globRegex(pattern);
      const results = walk(directory)
        .filter((path) => matches.test(path.split('/').at(-1)))
        .map((path) => path.replace(/^\/home(?=\/|$)/, '~'));
      this.write(results.join('\n') || `find: no matches found: ${pattern}`, results.length ? 'pth' : 'err');
    }

    showScript(path) {
      const target = path.includes('/') || this.currentDirectory === '/home/scripts'
        ? path
        : `/home/scripts/${path}`;
      const url = FILE_ROUTES.get(this.resolvePath(target));
      if (url) this.write(`curl -fsSL https://krasow.dev${url} | bash`, 'pth');
      else this.write(`show: no such script: ${path}`, 'err');
    }

    setTheme(requestedTheme) {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const theme = requestedTheme ?? (current === 'dark' ? 'light' : 'dark');
      if (!['light', 'dark'].includes(theme)) {
        this.write(`theme: unknown theme: ${theme}`, 'err');
        return;
      }
      document.documentElement.setAttribute('data-theme', theme);
      try { localStorage.setItem('theme', theme); } catch (error) {}
      this.write(`theme: ${theme}`, 'pth');
    }

    async readFile(path) {
      try {
        this.write(await this.fileText(path), 'pth');
      } catch (error) {
        const reason = error.message === 'EISDIR'
          ? 'is a directory'
          : error.message === 'ENOENT' ? 'no such text file' : 'unable to read file';
        this.write(`cat: ${path}: ${reason}`, 'err');
      }
    }

    async readFiles(paths) {
      for (const path of paths) {
        const files = this.expandPath(path);
        if (!files.length) {
          this.write(`cat: no matches found: ${path}`, 'err');
          continue;
        }
        for (const file of files) await this.readFile(file);
      }
    }

    async fileText(path) {
      if (DIRECTORIES[this.resolvePath(path)]) throw new Error('EISDIR');
      const source = this.textSource(path);
      if (!source) throw new Error('ENOENT');
      return this.loadText(source);
    }

    textSource(path) {
      const absolutePath = this.resolvePath(path);
      if (HIDDEN_FILES[absolutePath]) return { url: HIDDEN_FILES[absolutePath] };
      const homePath = absolutePath.replace(/^\/home\//, '');
      if (PAGE_SOURCES[homePath]) return PAGE_SOURCES[homePath];

      const url = READABLE_FILES[homePath]
        ?? (absolutePath.startsWith('/home/scripts/') ? FILE_ROUTES.get(absolutePath) : null);
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

    async grep(pattern, paths) {
      let expression;
      try {
        expression = new RegExp(pattern);
      } catch (error) {
        this.write(`grep: invalid pattern: ${pattern}`, 'err');
        return;
      }

      const files = paths.flatMap((path) => this.expandPath(path));
      if (!files.length) {
        this.write(`grep: no files matched`, 'err');
        return;
      }

      for (const path of files) {
        try {
          const matches = (await this.fileText(path))
            .split('\n')
            .filter((line) => expression.test(line));
          if (matches.length) this.write(matches.map((line) => `${path}:${line}`).join('\n'), 'pth');
        } catch (error) {
          const reason = error.message === 'EISDIR'
            ? 'is a directory'
            : error.message === 'ENOENT' ? 'no such text file' : 'unable to read file';
          this.write(`grep: ${path}: ${reason}`, 'err');
        }
      }
    }

    async copyFile(path) {
      try {
        const text = await this.fileText(path);
        try {
          await navigator.clipboard.writeText(text);
        } catch (error) {
          const textarea = makeElement('textarea', '', text);
          textarea.style.position = 'fixed';
          textarea.style.opacity = '0';
          document.body.append(textarea);
          textarea.select();
          const copied = document.execCommand('copy');
          textarea.remove();
          if (!copied) throw error;
        }
        this.write(`copied: ${path} to clipboard`, 'pth');
      } catch (error) {
        const reason = error.message === 'EISDIR'
          ? 'is a directory'
          : error.message === 'ENOENT' ? 'no such text file' : 'unable to copy file';
        this.write(`copy: ${path}: ${reason}`, 'err');
      }
    }

    async countFiles(paths) {
      const files = paths.flatMap((path) => this.expandPath(path));
      if (!files.length) {
        this.write('wc: no files matched', 'err');
        return;
      }
      for (const path of files) {
        try {
          const text = await this.fileText(path);
          const lines = text ? text.split('\n').length : 0;
          const words = text.trim() ? text.trim().split(/\s+/).length : 0;
          const characters = [...text].length;
          this.write(`${lines} ${words} ${characters} ${path}`, 'pth');
        } catch (error) {
          const reason = error.message === 'EISDIR'
            ? 'is a directory'
            : error.message === 'ENOENT' ? 'no such text file' : 'unable to read file';
          this.write(`wc: ${path}: ${reason}`, 'err');
        }
      }
    }

    pageText(section) {
      const copy = section.cloneNode(true);
      copy.querySelectorAll('.cv-date span + span').forEach((span) => span.before(' – '));
      copy.querySelectorAll('.pub-date br').forEach((breakElement) => breakElement.replaceWith(' – '));
      copy.querySelectorAll([
        'br', 'div', 'p', 'li', 'hr', 'h1', 'h2', 'h3',
        '.cv-title', '.cv-org', '.cv-desc',
      ].join(',')).forEach((element) => element.after('\n'));

      return copy.textContent
        .replace(/[ \t]+/g, ' ')
        .replace(/ *\n */g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
    }

    showTree() {
      const lines = ['.'];
      this.appendTreeEntries(lines, this.currentDirectory);
      this.write(lines.join('\n'), 'pth');
    }

    appendTreeEntries(lines, directory, prefix = '') {
      const entries = this.entriesIn(directory);
      entries.forEach((name, index) => {
        const last = index === entries.length - 1;
        const branch = last ? '└──' : '├──';
        lines.push(`${prefix}${branch} ${displayFile(name)}`);
        if (!name.endsWith('/')) return;
        const child = `${directory === '/' ? '' : directory}/${name.slice(0, -1)}`;
        this.appendTreeEntries(lines, child, `${prefix}${last ? '    ' : '│   '}`);
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

        const savedDirectory = saved.currentDirectory === ''
          ? '/home'
          : saved.currentDirectory;
        this.currentDirectory = this.entriesIn(savedDirectory)
          ? savedDirectory
          : '/home';
        const previous = saved.previousDirectory === '' ? '/home' : saved.previousDirectory;
        this.previousDirectory = typeof previous === 'string' && this.entriesIn(previous)
          ? previous
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
      if (event.key === ']' && !this.ui.input.value && this.chatSession) {
        event.preventDefault();
        this.toggleChatShell();
        return;
      }

      if (event.ctrlKey && event.key.toLowerCase() === 'c') {
        event.preventDefault();
        this.echo(`${this.ui.input.value}^C`);
        this.clearInput();
        if (this.chatMode) this.leaveChat();
        return;
      }

      const [command, target] = this.ui.input.value.trim().split(/\s+/);
      if (event.key === 'Enter' && command === 'open'
        && !this.ui.autocomplete.hidden && this.entriesIn(this.resolvePath(target))) {
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
      this.historyCursor = Math.max(
        0,
        Math.min(this.history.length, this.historyCursor + delta),
      );
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
      const pathCommand = /^(cat|cd|copy|find|grep|ls|open|show|wc)\b/.test(typed)
        ? typed.split(/\s/)[0]
        : !typed.includes(' ') ? '' : null;
      const paths = pathCommand !== null
        ? this.pathCompletions(path, pathCommand)
          .map((candidate) => `${typed.slice(0, tokenStart)}${candidate}`)
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
        candidates = this.completionEntries(this.currentDirectory, path)
          .map((name) => ({ name, path: this.resolvePath(name) }));
      } else {
        const directory = this.resolvePath(path.slice(0, separator) || '/');
        const prefix = path.slice(0, separator + 1);
        candidates = this.completionEntries(directory, path.slice(separator + 1))
          .map((name) => ({ name: `${prefix}${name}`, path: this.resolvePath(`${prefix}${name}`) }));
      }

      return candidates
        .filter(({ name }) => name.startsWith(path))
        .filter(({ path: candidate }) => {
          if (command === 'cd') return Boolean(DIRECTORIES[candidate]);
          if (command === 'open') {
            const openable = [...FILE_ROUTES.keys(), ...Object.keys(HIDDEN_FILES)];
            return openable.includes(candidate)
              || ((path.includes('/') || path.startsWith('.'))
                && openable.some((target) => target.startsWith(`${candidate}/`)));
          }
          if (!['cat', 'copy', 'grep', 'wc'].includes(command)) return true;
          return TEXT_PATHS.has(candidate)
            || ((path.includes('/') || path.startsWith('.'))
              && [...TEXT_PATHS].some((textPath) => textPath.startsWith(`${candidate}/`)));
        })
        .map(({ name }) => name);
    }

    completionEntries(directory, partial) {
      const visible = this.entriesIn(directory) ?? [];
      if (!partial.startsWith('.')) return visible;

      const prefix = directory === '/' ? '/' : `${directory}/`;
      const hidden = Object.keys(DIRECTORIES)
        .filter((path) => path.startsWith(`${prefix}.`))
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

      const active = this.completionCycle.index < 0
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
      const prefix = pathPrefix && matches.every((match) => match.startsWith(pathPrefix))
        ? pathPrefix
        : commandPrefix;
      const label = (match) => match.startsWith(prefix) ? match.slice(prefix.length) : match;
      this.showCompletions(matches.map(label), label(active));
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
    banner.dataset.t = 'terminal';
    banner.setAttribute('aria-label', 'krasow.dev terminal');
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
