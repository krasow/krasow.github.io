(() => {
  'use strict';

  class VirtualTrash {
    constructor({ directories, fileRoutes, storageKey }) {
      this.directories = directories;
      this.fileRoutes = fileRoutes;
      this.storageKey = storageKey;
      this.paths = this.load();
    }

    load() {
      try {
        const paths = JSON.parse(localStorage.getItem(this.storageKey));
        return new Set(
          Array.isArray(paths)
            ? paths.filter((path) => typeof path === 'string' && path.startsWith('/'))
            : [],
        );
      } catch (error) {
        return new Set();
      }
    }

    persist() {
      try {
        localStorage.setItem(this.storageKey, JSON.stringify([...this.paths]));
      } catch (error) {
        // Storage may be unavailable in private or restricted browser contexts.
      }
    }

    contains(path) {
      return [...this.paths].some((removed) => path === removed || path.startsWith(`${removed}/`));
    }

    exists(path) {
      if (this.contains(path)) return false;
      if (this.directories[path] || this.fileRoutes.has(path)) return true;
      const separator = path.lastIndexOf('/');
      const parent = path.slice(0, separator) || '/';
      const name = path.slice(separator + 1);
      return (this.directories[parent] ?? []).some((entry) => entry.replace(/\/$/, '') === name);
    }

    remove(path, { recursive }) {
      if (path !== '/home' && !path.startsWith('/home/')) return 'permission denied';
      if (!this.exists(path)) return 'no such file or directory';
      if (this.directories[path] && !recursive) return 'is a directory';
      this.paths.add(path);
      return null;
    }
  }

  class TerminalFiles {
    constructor(terminal, options) {
      this.terminal = terminal;
      Object.assign(this, options);
      [
        'resolvePath',
        'resolve',
        'matchingEntries',
        'expandPath',
        'entriesIn',
        'fileText',
        'textSource',
        'loadText',
      ].forEach((name) => {
        terminal[name] = this[name].bind(this);
      });
    }

    resolvePath(input) {
      let path = input.trim();
      if (!path || path === '~') return '/home';
      if (path.startsWith('~/')) path = `/home/${path.slice(2)}`;
      else if (!path.startsWith('/')) path = `${this.terminal.currentDirectory}/${path}`;
      const parts = [];
      path.split('/').forEach((part) => {
        if (!part || part === '.') return;
        if (part === '..') parts.pop();
        else parts.push(part);
      });
      return `/${parts.join('/')}`;
    }

    resolve(command) {
      const path = this.resolvePath(command);
      return (
        this.shortcuts[command] ??
        (this.terminal.trash.contains(path) ? null : this.fileRoutes.get(path))
      );
    }

    matchingEntries(path) {
      const separator = path.lastIndexOf('/');
      const pattern = path.slice(separator + 1);
      const directoryPath = separator < 0 ? '.' : path.slice(0, separator) || '/';
      const directory = this.resolvePath(directoryPath);
      const entries = this.entriesIn(directory);
      const prefix = separator < 0 ? '' : `${directory === '/' ? '' : directory}/`;
      if (!entries) return { directoryPath, prefix, matches: null };
      const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&');
      const expression = new RegExp(`^${escaped.replace(/\*/g, '.*').replace(/\?/g, '.')}$`);
      return { directoryPath, prefix, matches: entries.filter((entry) => expression.test(entry)) };
    }

    expandPath(path) {
      if (!/[*?]/.test(path)) return [path];
      const { prefix, matches } = this.matchingEntries(path);
      return (matches ?? []).map((name) => `${prefix}${name}`);
    }

    entriesIn(directory) {
      if (this.terminal.trash.contains(directory) || !this.directories[directory]) return null;
      return this.directories[directory]
        .filter((name) => {
          if (name.startsWith('.')) return false;
          const child = `${directory === '/' ? '' : directory}/${name.replace(/\/$/, '')}`;
          return !this.terminal.trash.contains(child);
        })
        .sort((left, right) =>
          left.replace(/\/$/, '').localeCompare(right.replace(/\/$/, ''), undefined, {
            numeric: true,
            sensitivity: 'base',
          }),
        );
    }

    async fileText(path) {
      const resolved = this.resolvePath(path);
      if (this.terminal.trash.contains(resolved)) throw new Error('ENOENT');
      if (this.directories[resolved]) throw new Error('EISDIR');
      const source = this.textSource(path);
      if (!source) throw new Error('ENOENT');
      return this.loadText(source);
    }

    textSource(path) {
      const absolutePath = this.resolvePath(path);
      if (this.terminal.trash.contains(absolutePath)) return null;
      // `.pg` pages resolve to committed, pre-parsed text (see
      // terminal/lib/page-text.mjs); plain text files resolve to themselves.
      const url = this.textPaths.has(absolutePath) ? this.textRoutes.get(absolutePath) : null;
      return url ? { url } : null;
    }

    async loadText(source) {
      const response = await fetch(source.url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return (await response.text()).trim();
    }
  }

  const addEntry = (directories, directory, entry) => {
    directories[directory] ??= [];
    if (!directories[directory].includes(entry)) directories[directory].push(entry);
  };

  const addDirectory = (directories, path) => {
    const parts = path.split('/').filter(Boolean);
    let parent = '/';
    parts.forEach((part) => {
      addEntry(directories, parent, `${part}/`);
      parent = `${parent === '/' ? '' : parent}/${part}`;
      directories[parent] ??= [];
    });
  };

  const hydrate = (files, { directories, fileRoutes, textPaths, textRoutes }, folders = []) => {
    folders.forEach((folder) => addDirectory(directories, folder));
    files.forEach(({ path, url, target, text = true }) => {
      const parts = path.split('/').filter(Boolean);
      let directory = '/';
      parts.slice(0, -1).forEach((part) => {
        addEntry(directories, directory, `${part}/`);
        directory = `${directory === '/' ? '' : directory}/${part}`;
        directories[directory] ??= [];
      });

      addEntry(directories, directory, parts.at(-1));
      // `target` (if any) is where `open` navigates; `url` is the text content.
      fileRoutes.set(path, target ?? url);
      if (text) {
        textPaths.add(path);
        textRoutes.set(path, url);
      }
    });
  };

  const loadManifest = async (url, fileSystem) => {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { directories = [], files } = await response.json();
    if (!Array.isArray(files)) throw new Error('Invalid filesystem manifest');
    hydrate(files, fileSystem, directories);
  };

  window.KrasowTerminalFileSystem = { TerminalFiles, VirtualTrash, hydrate, loadManifest };
})();
