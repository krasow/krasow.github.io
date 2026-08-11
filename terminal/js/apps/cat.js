(() => {
  'use strict';
  class CatApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      if (!args.length) return false;
      this.read(args);
      return true;
    }
    async read(paths) {
      for (const path of paths) {
        const files = this.terminal.expandPath(path);
        if (!files.length) {
          this.terminal.write(`cat: no matches found: ${path}`, 'err');
          continue;
        }
        for (const file of files) {
          try {
            this.terminal.write(await this.terminal.fileText(file), 'pth');
          } catch (error) {
            this.terminal.write(`cat: ${file}: ${window.TerminalApp.readableError(error)}`, 'err');
          }
        }
      }
    }
  }
  window.CatApp = CatApp;
})();
