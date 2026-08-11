(() => {
  'use strict';
  class OpenApp {
    constructor(terminal, { shortcuts, fileRoutes }) {
      this.terminal = terminal;
      this.shortcuts = shortcuts;
      this.fileRoutes = fileRoutes;
    }
    run(args) {
      return window.TerminalApp.exact(args, 1, () => this.open(args[0]));
    }
    open(target, allowDownload = false) {
      const path = this.terminal.resolvePath(target);
      if (path.endsWith('.vcf') && !allowDownload) {
        this.terminal.write(`open: ${target}: use \`download ${target}\` for contact cards`, 'err');
        return;
      }
      if (this.terminal.entriesIn(path)) {
        this.terminal.write(`open: ${target}: is a directory`, 'err');
        return;
      }
      const url =
        this.shortcuts[target] ??
        (this.terminal.trash.contains(path) ? null : this.fileRoutes.get(path));
      if (url) this.terminal.navigate(url);
      else this.terminal.write(`open: ${target}: no such file or page`, 'err');
    }
  }
  window.OpenApp = OpenApp;
})();
