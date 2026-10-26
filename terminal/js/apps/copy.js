(() => {
  'use strict';
  class CopyApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      if (!args.length && this.terminal.stdin !== undefined)
        return this.copy('piped input', async () => this.terminal.stdin).then(() => true);
      if (args.length !== 1) return false;
      return this.copy(args[0], () => this.terminal.fileText(args[0])).then(() => true);
    }
    async copy(path, read) {
      try {
        const text = await read();
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
