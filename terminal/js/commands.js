(() => {
  'use strict';

  const hiddenResponses = { hi: 'Hello!', hello: 'Hello!' };
  const easterEggs = new Map([
    ['sudo make me a sandwich', 'Okay. [sandwich delivered]'],
    ['make me a sandwich', 'What? Make it yourself.'],
    ['42', 'The answer to life, the universe, and distributed computing.'],
    ['coffee', 'Error: coffee machine is not attached to this runtime.'],
    ['fortune', 'Parallelism is easy. Scheduling it is the research project.'],
    [
      'neofetch',
      [
        '       /\\       david@krasow.dev',
        '      /  \\      ----------------',
        '     / /\\ \\     OS: krasow.dev',
        '    / ____ \\    Shell: zsh',
        '   /_/    \\_\\   Runtime: Legion',
      ].join('\n'),
    ],
  ]);
  const usage = {
    chat: 'chat [question]',
    snake: 'snake',
    clear: 'clear [-x]',
    help: 'help',
    pwd: 'pwd',
    tree: 'tree',
    theme: 'theme [light|dark]',
    cat: 'cat <file|pattern> [...]  (or read piped input)',
    ls: 'ls [folder|pattern]',
    cd: 'cd [folder|..|-]',
    show: 'show <script>',
    echo: 'echo [text]',
    cowsay: 'cowsay [text]',
    grep: 'grep [-i] <pattern> <file|pattern> [...]  (or read piped input)',
    copy: 'copy <file>  (or copy piped input)',
    wc: 'wc <file|pattern> [...]  (or count piped input)',
    rm: 'rm [-rf] <file|folder> [...]',
    open: 'open <file|page>',
    download: 'download <file>',
    find: 'find [folder] [pattern]',
    reset: 'reset',
    sudo: 'sudo <command>',
    whoami: 'whoami',
    exit: 'exit',
  };
  const help = [
    [
      'Navigation',
      [
        ['ls [folder|pattern]', 'list pages or matching files'],
        ['cd folder · cd .. · cd -', 'change directory'],
        ['pwd · tree', 'inspect the current directory'],
        ['find [folder] [pattern]', 'find files recursively'],
        ['open file|page', 'open a page or document'],
        ['download file', 'download a file'],
        ['cat file|pattern', 'read one or more text files'],
        ['grep pattern file', 'search one or more text files'],
        ['cmd | grep pattern', 'pipe output into grep, wc, cat, or copy'],
        ['copy file', 'copy a text file'],
        ['wc file|pattern', 'count lines, words, and characters'],
        ['rm [-rf] file|folder', 'hide entries from the virtual filesystem'],
        ['echo text', 'print text'],
        ['show script', 'print an install command'],
      ],
    ],
    [
      'Information',
      [
        ['chat [question]', 'ask about David or enter chat mode'],
        ['whoami', 'show name'],
        ['cat contact.md', 'show contact details'],
      ],
    ],
    ['Games', [['snake', 'play Snake in the terminal']]],
    [
      'Links',
      [
        ['github · zoom', 'open an external page'],
        ['resume.pdf · contact.vcf', 'open or download a document'],
      ],
    ],
    [
      'Controls',
      [
        ['clear', 'clear the terminal and scrollback'],
        ['clear -x', 'clear the screen but preserve scrollback'],
        ['theme [light|dark]', 'change color theme'],
        ['reset', 'restore all locally saved terminal state'],
        ['Esc · Ctrl+C', 'cancel current input'],
        ['↑ / ↓ · Tab', 'history and autocomplete'],
      ],
    ],
  ];

  class TerminalCommands {
    constructor(terminal, options) {
      this.terminal = terminal;
      this.shortcuts = options.shortcuts;
      this.directories = options.directories;
      this.apps = {
        cat: new window.CatApp(terminal),
        cd: new window.CdApp(terminal),
        chat: new window.ChatApp(terminal),
        clear: new window.ClearApp(terminal),
        copy: new window.CopyApp(terminal),
        cowsay: new window.CowsayApp(terminal),
        find: new window.FindApp(terminal),
        grep: new window.GrepApp(terminal),
        help: new window.HelpApp(terminal, help),
        ls: new window.LsApp(terminal),
        open: new window.OpenApp(terminal, options),
        reset: new window.ResetApp(terminal),
        rm: new window.RmApp(terminal),
        show: new window.ShowApp(terminal, options.fileRoutes),
        snake: new window.SnakeApp(terminal),
        theme: new window.ThemeApp(terminal),
        tree: new window.TreeApp(terminal),
        wc: new window.WcApp(terminal),
      };
      this.apps.download = new window.DownloadApp(terminal, this.apps.open);
      this.handlers = new Map(
        Object.entries(this.apps).map(([name, app]) => [name, (args) => app.run(args)]),
      );
      this.handlers.set('pwd', (args) =>
        window.TerminalApp.exact(args, 0, () => terminal.write(terminal.path(), 'pth')),
      );
      this.handlers.set('echo', (args) => {
        terminal.write(args.join(' '), 'pth');
        return true;
      });
      this.handlers.set('sudo', () => {
        terminal.write('david is not in the sudoers file. This incident will be reported.', 'err');
        return true;
      });
      this.handlers.set('whoami', (args) =>
        window.TerminalApp.exact(args, 0, () => terminal.write('David Krasowska', 'pth')),
      );
      this.handlers.set('exit', (args) => window.TerminalApp.exact(args, 0, () => terminal.exit()));
    }

    commands() {
      return this.handlers;
    }

    completions() {
      return [
        ...new Set([
          ...Object.keys(usage),
          'cd ..',
          'cd -',
          'theme light',
          'theme dark',
          'clear -x',
          ...(this.directories['/home/scripts'] ?? []).map((name) => `show ${name}`),
          ...Object.keys(this.shortcuts),
        ]),
      ];
    }

    execute(command) {
      if (command.includes('|')) {
        this.pipeline(command);
        return true;
      }
      const easterEgg = easterEggs.get(command.toLowerCase());
      if (easterEgg) {
        this.terminal.write(easterEgg, 'pth');
        return true;
      }
      const [name, ...args] = command.split(/\s+/);
      const handler = this.handlers.get(name);
      if (handler) {
        if (!handler(args)) this.terminal.write(`${name}: usage: ${usage[name]}`, 'err');
        return true;
      }
      const response = hiddenResponses[name];
      if (response && !args.length) {
        this.terminal.write(response, 'pth');
        return true;
      }
      return false;
    }

    // `a | b | c`: each stage's output becomes the next stage's input
    // (`terminal.stdin`). Errors from any stage still print immediately.
    pipeline(command) {
      const stages = command.split('|').map((stage) => stage.trim());
      if (stages.some((stage) => !stage)) {
        this.terminal.write("zsh: parse error near `|'", 'err');
        return;
      }
      this.running = this.runPipeline(stages).finally(() => (this.running = null));
    }

    async runPipeline(stages) {
      let input;
      for (const [index, stage] of stages.entries()) {
        const [name, ...args] = stage.split(/\s+/);
        const handler = this.handlers.get(name);
        if (!handler) {
          this.terminal.write(`zsh: command not found: ${name}`, 'err');
          return;
        }
        const output = index === stages.length - 1 ? null : [];
        const restore = this.redirect(output, input);
        try {
          if (!(await handler(args))) this.terminal.write(`${name}: usage: ${usage[name]}`, 'err');
        } finally {
          restore();
        }
        input = output?.join('\n');
      }
    }

    // Route the terminal's output into `output` (unless null) and expose
    // `stdin` to apps; returns a function that undoes both.
    redirect(output, stdin) {
      const terminal = this.terminal;
      const keys = ['write', 'writeListing', 'writeLink'];
      const saved = keys.map((key) => Object.getOwnPropertyDescriptor(terminal, key));
      const write = terminal.write.bind(terminal);
      terminal.stdin = stdin;
      if (output) {
        terminal.write = (text, className = '') =>
          className === 'err' ? write(text, className) : output.push(String(text));
        terminal.writeListing = (entries) => output.push(...entries);
        terminal.writeLink = (url, text) => output.push(text);
      }
      return () => {
        delete terminal.stdin;
        keys.forEach((key, index) =>
          saved[index] ? Object.defineProperty(terminal, key, saved[index]) : delete terminal[key],
        );
      };
    }
  }

  window.KrasowTerminalCommands = { TerminalCommands };
})();
