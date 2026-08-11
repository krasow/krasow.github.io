(() => {
  'use strict';
  class ShowApp {
    constructor(terminal, fileRoutes) {
      this.terminal = terminal;
      this.fileRoutes = fileRoutes;
    }
    run(args) {
      return window.TerminalApp.exact(args, 1, () => this.show(args[0]));
    }
    show(path) {
      const target =
        path.includes('/') || this.terminal.currentDirectory === '/home/scripts'
          ? path
          : `/home/scripts/${path}`;
      const resolved = this.terminal.resolvePath(target);
      const url = this.terminal.trash.contains(resolved) ? null : this.fileRoutes.get(resolved);
      if (url) this.terminal.write(`curl -fsSL https://krasow.dev${url} | bash`, 'pth');
      else this.terminal.write(`show: no such script: ${path}`, 'err');
    }
  }
  window.ShowApp = ShowApp;
})();
