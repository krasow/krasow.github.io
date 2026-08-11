(() => {
  'use strict';
  class LsApp {
    constructor(terminal) {
      this.terminal = terminal;
    }
    run(args) {
      this.list(args.join(' '));
      return true;
    }
    list(path) {
      if (/[*?]/.test(path)) {
        const { directoryPath, matches } = this.terminal.matchingEntries(path);
        if (matches === null) this.terminal.write(`ls: ${directoryPath}: not a directory`, 'err');
        else if (matches.length) this.terminal.writeListing(matches);
        else this.terminal.write(`ls: no matches found: ${path}`, 'err');
        return;
      }
      const directory = path ? this.terminal.resolvePath(path) : this.terminal.currentDirectory;
      const entries = this.terminal.entriesIn(directory);
      if (!entries) this.terminal.write(`ls: ${path}: not a directory`, 'err');
      else this.terminal.writeListing(entries);
    }
  }
  window.LsApp = LsApp;
})();
