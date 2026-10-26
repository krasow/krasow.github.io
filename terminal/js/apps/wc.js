(() => {
  'use strict';

  // "lines words characters"
  const counts = (text) => {
    const lines = text ? text.split('\n').length : 0;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    return `${lines} ${words} ${[...text].length}`;
  };

  class WcApp {
    constructor(terminal) {
      this.terminal = terminal;
    }

    run(args) {
      if (!args.length) {
        if (this.terminal.stdin === undefined) return false;
        this.terminal.write(counts(this.terminal.stdin), 'pth');
        return true;
      }
      return this.count(args).then(() => true);
    }

    async count(paths) {
      const files = paths.flatMap((path) => this.terminal.expandPath(path));
      if (!files.length) {
        this.terminal.write('wc: no files matched', 'err');
        return;
      }

      for (const path of files) {
        try {
          const text = await this.terminal.fileText(path);
          this.terminal.write(`${counts(text)} ${path}`, 'pth');
        } catch (error) {
          const reason =
            error.message === 'EISDIR'
              ? 'is a directory'
              : error.message === 'ENOENT'
                ? 'no such text file'
                : 'unable to read file';
          this.terminal.write(`wc: ${path}: ${reason}`, 'err');
        }
      }
    }
  }

  window.WcApp = WcApp;
})();
