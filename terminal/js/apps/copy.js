(() => {
  'use strict';
  class CopyApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      return window.TerminalApp.exact(args, 1, () => this.copy(args[0]));
    }
    async copy(path) {
      try {
        const text = await this.terminal.fileText(path);
        try {
          await navigator.clipboard.writeText(text);
        } catch (error) {
          const textarea = document.createElement('textarea');
          textarea.textContent = text;
          Object.assign(textarea.style, { position: 'fixed', opacity: '0' });
          document.body.append(textarea);
          textarea.select();
          const copied = document.execCommand('copy');
          textarea.remove();
          if (!copied) throw error;
        }
        this.terminal.write(`copied: ${path} to clipboard`, 'pth');
      } catch (error) {
        this.terminal.write(
          `copy: ${path}: ${window.TerminalApp.readableError(error, 'unable to copy file')}`,
          'err',
        );
      }
    }
  }
  window.CopyApp = CopyApp;
})();
