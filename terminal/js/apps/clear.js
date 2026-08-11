(() => {
  'use strict';
  class ClearApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      if (!args.length) this.terminal.clearLog();
      else if (args.length === 1 && args[0] === '-x') this.terminal.clearScreen();
      else return false;
      return true;
    }
  }
  window.ClearApp = ClearApp;
})();
