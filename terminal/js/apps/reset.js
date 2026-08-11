(() => {
  'use strict';
  class ResetApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      return window.TerminalApp.exact(args, 0, () => this.reset());
    }
    reset() {
      try {
        localStorage.clear();
      } catch (error) {}
      const line = document.createElement('p');
      line.className = 'ln hint';
      line.textContent = 'resetting local terminal state…';
      this.terminal.append(line);
      setTimeout(() => location.reload(), 250);
    }
  }
  window.ResetApp = ResetApp;
})();
