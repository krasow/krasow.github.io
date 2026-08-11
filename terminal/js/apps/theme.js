(() => {
  'use strict';
  class ThemeApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      return window.TerminalApp.maximum(args, 1, () => this.set(args[0]));
    }
    set(requested) {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const theme = requested ?? (current === 'dark' ? 'light' : 'dark');
      if (!['light', 'dark'].includes(theme)) {
        this.terminal.write(`theme: unknown theme: ${theme}`, 'err');
        return;
      }
      document.documentElement.setAttribute('data-theme', theme);
      try {
        localStorage.setItem('theme', theme);
      } catch (error) {}
      this.terminal.write(`theme: ${theme}`, 'pth');
    }
  }
  window.ThemeApp = ThemeApp;
})();
