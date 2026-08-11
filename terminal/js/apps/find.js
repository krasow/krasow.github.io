(() => {
  'use strict';
  class FindApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      return window.TerminalApp.maximum(args, 2, () => this.find(args));
    }
    find(args) {
      const hasDirectory =
        args.length === 2 ||
        (args[0] && this.terminal.entriesIn(this.terminal.resolvePath(args[0])));
      const directory = hasDirectory
        ? this.terminal.resolvePath(args[0])
        : this.terminal.currentDirectory;
      const pattern = args[hasDirectory ? 1 : 0] ?? '*';
      if (!this.terminal.entriesIn(directory)) {
        this.terminal.write(`find: ${args[0]}: not a directory`, 'err');
        return;
      }
      const walk = (folder) =>
        this.terminal.entriesIn(folder).flatMap((name) => {
          const path = `${folder === '/' ? '' : folder}/${name.replace(/\/$/, '')}`;
          return [path, ...(name.endsWith('/') ? walk(path) : [])];
        });
      const matches = window.TerminalApp.globRegex(pattern);
      const results = walk(directory)
        .filter((path) => matches.test(path.split('/').at(-1)))
        .map((path) => path.replace(/^\/home(?=\/|$)/, '~'));
      this.terminal.write(
        results.join('\n') || `find: no matches found: ${pattern}`,
        results.length ? 'pth' : 'err',
      );
    }
  }
  window.FindApp = FindApp;
})();
