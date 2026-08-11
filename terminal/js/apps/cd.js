(() => {
  'use strict';
  class CdApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      return window.TerminalApp.maximum(args, 1, () => this.change(args[0] ?? ''));
    }
    change(target) {
      const requested = target === '-' ? '-' : this.terminal.resolvePath(target);
      let next = requested;
      if (requested === '-') {
        if (this.terminal.previousDirectory === null) {
          this.terminal.write('cd: no previous directory', 'err');
          return;
        }
        next = this.terminal.previousDirectory;
      } else if (!this.terminal.entriesIn(requested)) {
        this.terminal.write(`cd: no such file or directory: ${target}`, 'err');
        return;
      }
      this.terminal.previousDirectory = this.terminal.currentDirectory;
      this.terminal.currentDirectory = next;
      this.terminal.updatePrompt();
      this.terminal.persist();
    }
  }
  window.CdApp = CdApp;
})();
