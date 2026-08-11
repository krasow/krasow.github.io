(() => {
  'use strict';

  class TerminalAutocomplete {
    constructor(terminal, { directories, fileRoutes, hiddenFiles, textPaths }) {
      this.terminal = terminal;
      this.directories = directories;
      this.fileRoutes = fileRoutes;
      this.hiddenFiles = hiddenFiles;
      this.textPaths = textPaths;
      this.commands = terminal.commandSet.completions();
      this.cycle = null;
    }

    clear() {
      this.terminal.ui.input.value = '';
      this.hide();
    }

    accept() {
      if (this.terminal.ui.autocomplete.hidden) return false;
      const [command, ...args] = this.terminal.ui.input.value.trim().split(/\s+/);
      const token = args.at(-1) ?? command;
      const entries = this.terminal.entriesIn(this.terminal.resolvePath(token));
      if (!entries) return false;
      if (command === 'cd' && !entries.some((entry) => entry.endsWith('/'))) return false;
      this.hide();
      return true;
    }

    recall(delta) {
      const terminal = this.terminal;
      terminal.historyCursor = Math.max(
        0,
        Math.min(terminal.history.length, terminal.historyCursor + delta),
      );
      terminal.ui.input.value = terminal.history[terminal.historyCursor] ?? '';
      this.hide();
    }

    complete() {
      const input = this.terminal.ui.input;
      const typed = input.value.trimStart();
      if (!typed.trim()) return this.hide();
      if (this.cycle?.value === typed) return this.cycleNext();

      const tokenStart = typed.lastIndexOf(' ') + 1;
      const path = typed.slice(tokenStart);
      const pathCommand = /^(cat|cd|copy|download|find|grep|ls|open|rm|show|wc)\b/.test(typed)
        ? typed.split(/\s/)[0]
        : !typed.includes(' ')
          ? ''
          : null;
      const paths =
        pathCommand === null
          ? []
          : this.pathMatches(path, pathCommand).map(
              (candidate) => `${typed.slice(0, tokenStart)}${candidate}`,
            );
      const matches = [...new Set([...this.commands, ...paths])]
        .filter((command) => command.startsWith(typed))
        .sort((a, b) => a.localeCompare(b));

      if (!matches.length) this.hide();
      else if (matches.length === 1) {
        input.value = matches[0];
        this.hide();
      } else this.completeMultiple(typed, matches);
    }

    pathMatches(path, command) {
      const separator = path.lastIndexOf('/');
      let candidates;
      if (separator < 0) {
        candidates = this.entries(this.terminal.currentDirectory, path).map((name) => ({
          name,
          path: this.terminal.resolvePath(name),
        }));
      } else {
        const directory = this.terminal.resolvePath(path.slice(0, separator) || '/');
        const prefix = path.slice(0, separator + 1);
        candidates = this.entries(directory, path.slice(separator + 1)).map((name) => ({
          name: `${prefix}${name}`,
          path: this.terminal.resolvePath(`${prefix}${name}`),
        }));
      }
      return candidates
        .filter(({ name }) => name.startsWith(path))
        .filter(({ path: candidate }) => this.accepts(candidate, path, command))
        .map(({ name }) => name);
    }

    accepts(candidate, typedPath, command) {
      if (command === 'cd') return Boolean(this.directories[candidate]);
      if (command === 'open' || command === 'download') {
        const openable = [...this.fileRoutes.keys(), ...Object.keys(this.hiddenFiles)];
        const matches =
          openable.includes(candidate) ||
          ((typedPath.includes('/') || typedPath.startsWith('.')) &&
            openable.some((path) => path.startsWith(`${candidate}/`)));
        if (!matches || (command === 'open' && /\.vcf$/i.test(candidate))) return false;
        return (
          command !== 'download' ||
          this.directories[candidate] ||
          /\.(pdf|sh|vcf)$/i.test(candidate)
        );
      }
      if (!['cat', 'copy', 'grep', 'wc'].includes(command)) return true;
      return (
        this.textPaths.has(candidate) ||
        ((typedPath.includes('/') || typedPath.startsWith('.')) &&
          [...this.textPaths].some((path) => path.startsWith(`${candidate}/`)))
      );
    }

    entries(directory, partial) {
      const visible = this.terminal.entriesIn(directory) ?? [];
      if (!partial.startsWith('.')) return visible;
      const prefix = directory === '/' ? '/' : `${directory}/`;
      const hidden = Object.keys(this.directories)
        .filter((path) => path.startsWith(`${prefix}.`))
        .filter((path) => !this.terminal.trash.contains(path))
        .map((path) => `${path.slice(prefix.length).split('/')[0]}/`);
      return [...new Set([...visible, ...hidden])];
    }

    completeMultiple(typed, matches) {
      const prefix = matches.reduce((common, match) => {
        let end = 0;
        while (end < common.length && common[end] === match[end]) end += 1;
        return common.slice(0, end);
      });
      const value = prefix.length > typed.length ? prefix : matches[0];
      this.terminal.ui.input.value = value;
      this.cycle = { matches, index: matches.indexOf(value), value };
      this.showMenu(matches, this.cycle.index < 0 ? '' : value);
    }

    cycleNext() {
      this.cycle.index = (this.cycle.index + 1) % this.cycle.matches.length;
      this.cycle.value = this.cycle.matches[this.cycle.index];
      this.terminal.ui.input.value = this.cycle.value;
      this.showMenu(this.cycle.matches, this.cycle.value);
    }

    showMenu(matches, active) {
      const commandPrefix = `${matches[0].split(' ')[0]} `;
      const slash = matches[0].lastIndexOf('/');
      const pathPrefix = slash < 0 ? '' : matches[0].slice(0, slash + 1);
      const prefix =
        pathPrefix && matches.every((match) => match.startsWith(pathPrefix))
          ? pathPrefix
          : commandPrefix;
      const label = (match) => (match.startsWith(prefix) ? match.slice(prefix.length) : match);
      this.show(matches.map(label), label(active));
    }

    show(choices, active = '') {
      const element = this.terminal.ui.autocomplete;
      element.replaceChildren(
        ...choices.map((choice) => {
          const item = document.createElement('span');
          item.className = `autocomplete-choice${choice === active ? ' active' : ''}`;
          item.textContent = choice;
          return item;
        }),
      );
      element.hidden = !choices.length;
    }

    hide() {
      this.cycle = null;
      this.terminal.ui.autocomplete.hidden = true;
      this.terminal.ui.autocomplete.replaceChildren();
    }
  }

  window.TerminalAutocomplete = TerminalAutocomplete;
})();
