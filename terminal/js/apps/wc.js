(() => {
  'use strict';

  class WcApp {
    constructor(terminal) {
      this.terminal = terminal;
    }

    run(args) {
      if (!args.length) return false;
      this.count(args);
      return true;
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
          const lines = text ? text.split('\n').length : 0;
          const words = text.trim() ? text.trim().split(/\s+/).length : 0;
          const characters = [...text].length;
          this.terminal.write(`${lines} ${words} ${characters} ${path}`, 'pth');
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
