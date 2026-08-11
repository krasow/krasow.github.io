(() => {
  'use strict';
  class TreeApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      return window.TerminalApp.exact(args, 0, () => this.show());
    }
    show() {
      const lines = ['.'];
      this.append(lines, this.terminal.currentDirectory);
      this.terminal.write(lines.join('\n'), 'pth');
    }
    append(lines, directory, prefix = '') {
      const entries = this.terminal.entriesIn(directory);
      entries.forEach((name, index) => {
        const last = index === entries.length - 1;
        lines.push(`${prefix}${last ? '└──' : '├──'} ${this.terminal.displayFile(name)}`);
        if (!name.endsWith('/')) return;
        const child = `${directory === '/' ? '' : directory}/${name.slice(0, -1)}`;
        this.append(lines, child, `${prefix}${last ? '    ' : '│   '}`);
      });
    }
  }
  window.TreeApp = TreeApp;
})();
