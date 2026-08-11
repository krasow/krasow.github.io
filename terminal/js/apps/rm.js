(() => {
  'use strict';
  class RmApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      let recursive = false;
      let force = false;
      const targets = [];
      for (const arg of args) {
        if (arg.startsWith('-') && arg.length > 1 && !targets.length) {
          const flags = arg.slice(1);
          if (/[^rf]/.test(flags)) return false;
          recursive ||= flags.includes('r');
          force ||= flags.includes('f');
        } else targets.push(arg);
      }
      if (!targets.length) return false;
      for (const target of targets) {
        const expanded = /[*?]/.test(target) ? this.terminal.expandPath(target) : [target];
        if (!expanded.length && !force)
          this.terminal.write(`rm: ${target}: no matches found`, 'err');
        for (const item of expanded) {
          const path = this.terminal.resolvePath(item);
          const error = this.terminal.trash.remove(path, { recursive });
          if (error && (!force || error !== 'no such file or directory')) {
            this.terminal.write(`rm: ${item}: ${error}`, 'err');
          } else if (
            !error &&
            (path === this.terminal.currentDirectory ||
              this.terminal.currentDirectory.startsWith(`${path}/`))
          ) {
            this.terminal.currentDirectory = path.slice(0, path.lastIndexOf('/')) || '/';
            this.terminal.previousDirectory = null;
            this.terminal.updatePrompt();
            this.terminal.persist();
          }
        }
      }
      this.terminal.trash.persist();
      return true;
    }
  }
  window.RmApp = RmApp;
})();
