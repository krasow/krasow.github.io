(() => {
  'use strict';

  class GrepApp {
    constructor(terminal) {
      this.terminal = terminal;
    }

    run(args) {
      const ignoreCase = args[0] === '-i';
      const offset = ignoreCase ? 1 : 0;
      const piped = this.terminal.stdin !== undefined && args.length === offset + 1;
      if (args.length < offset + 2 && !piped) return false;

      const expression = this.expression(args[offset], ignoreCase);
      if (!expression) return true;
      if (piped) {
        const matches = this.terminal.stdin.split('\n').filter((line) => expression.test(line));
        if (matches.length) this.terminal.write(matches.join('\n'), 'pth');
        return true;
      }
      return this.search(expression, args.slice(offset + 1)).then(() => true);
    }

    expression(pattern, ignoreCase) {
      try {
        return new RegExp(pattern, ignoreCase ? 'i' : '');
      } catch (error) {
        this.terminal.write(`grep: invalid pattern: ${pattern}`, 'err');
        return null;
      }
    }

    async search(expression, paths) {
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
