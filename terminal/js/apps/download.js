(() => {
  'use strict';
  class DownloadApp {
    constructor(terminal, openApp) {
      this.terminal = terminal;
      this.openApp = openApp;
    }
    run(args) {
      return window.TerminalApp.exact(args, 1, () => this.download(args[0]));
    }
    download(target) {
      if (!/\.(pdf|sh|vcf)$/i.test(this.terminal.resolvePath(target))) {
        this.terminal.write(`download: ${target}: not a downloadable document`, 'err');
        return;
      }
      this.openApp.open(target, true);
    }
  }
  window.DownloadApp = DownloadApp;
})();
