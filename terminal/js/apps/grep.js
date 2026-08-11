(() => {
  'use strict';

  class GrepApp {
    constructor(terminal) {
      this.terminal = terminal;
    }

    run(args) {
      const ignoreCase = args[0] === '-i';
      const offset = ignoreCase ? 1 : 0;
      if (args.length < offset + 2) return false;

      this.search(args[offset], args.slice(offset + 1), ignoreCase);
      return true;
    }

    async search(pattern, paths, ignoreCase = false) {
      let expression;
      try {
        expression = new RegExp(pattern, ignoreCase ? 'i' : '');
      } catch (error) {
        this.terminal.write(`grep: invalid pattern: ${pattern}`, 'err');
        return;
      }

      const files = paths.flatMap((path) => this.terminal.expandPath(path));
      if (!files.length) {
        this.terminal.write('grep: no files matched', 'err');
        return;
      }

      for (const path of files) {
        try {
          const matches = (await this.terminal.fileText(path))
            .split('\n')
            .filter((line) => expression.test(line));
          if (matches.length) {
            this.terminal.write(matches.map((line) => `${path}:${line}`).join('\n'), 'pth');
          }
        } catch (error) {
          const reason =
            error.message === 'EISDIR'
              ? 'is a directory'
              : error.message === 'ENOENT'
                ? 'no such text file'
                : 'unable to read file';
          this.terminal.write(`grep: ${path}: ${reason}`, 'err');
        }
      }
    }
  }

  window.GrepApp = GrepApp;
})();
