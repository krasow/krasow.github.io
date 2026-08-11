(() => {
  'use strict';

  class VirtualTrash {
    constructor({ directories, fileRoutes, hiddenFiles, storageKey }) {
      this.directories = directories;
      this.fileRoutes = fileRoutes;
      this.hiddenFiles = hiddenFiles;
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
      if (this.directories[path] || this.fileRoutes.has(path) || this.hiddenFiles[path])
        return true;
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
        'pageText',
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
      return this.directories[directory].filter((name) => {
        if (name.startsWith('.')) return false;
        const child = `${directory === '/' ? '' : directory}/${name.replace(/\/$/, '')}`;
        return !this.terminal.trash.contains(child);
      });
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
      if (this.hiddenFiles[absolutePath]) return { url: this.hiddenFiles[absolutePath] };
      const homePath = absolutePath.replace(/^\/home\//, '');
      if (this.pageSources[homePath]) return this.pageSources[homePath];
      const url =
        this.readableFiles[homePath] ??
        (this.textPaths.has(absolutePath) ? this.fileRoutes.get(absolutePath) : null);
      return url ? { url } : null;
    }

    async loadText(source) {
      const response = await fetch(source.url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const content = (await response.text()).trim();
      if (!source.selector) return content;
      const document = new DOMParser().parseFromString(content, 'text/html');
      const text = [...document.querySelectorAll(source.selector)]
        .map((section) => this.pageText(section))
        .filter(Boolean)
        .join('\n\n')
        .replace(/\n{3,}/g, '\n\n');
      if (!text) throw new Error('No readable content');
      return text;
    }

    pageText(section) {
      const copy = section.cloneNode(true);
      copy.querySelectorAll('.cv-date span + span').forEach((span) => span.before(' – '));
      copy.querySelectorAll('.pub-date br').forEach((br) => br.replaceWith(' – '));
      copy
        .querySelectorAll(
          ['br', 'div', 'p', 'li', 'hr', 'h1', 'h2', 'h3', '.cv-title', '.cv-org', '.cv-desc'].join(
            ',',
          ),
        )
        .forEach((element) => element.after('\n'));
      return copy.textContent
        .replace(/[ \t]+/g, ' ')
        .replace(/ *\n */g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
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

  const hydrate = (
    files,
    { directories, fileRoutes, pageSources = {}, textPaths },
    folders = [],
  ) => {
    folders.forEach((folder) => addDirectory(directories, folder));
    files.forEach(({ path, url, target, source, selector = '.holder', text = true }) => {
      const parts = path.split('/').filter(Boolean);
      let directory = '/';
      parts.slice(0, -1).forEach((part) => {
        addEntry(directories, directory, `${part}/`);
        directory = `${directory === '/' ? '' : directory}/${part}`;
        directories[directory] ??= [];
      });

      addEntry(directories, directory, parts.at(-1));
      fileRoutes.set(path, target ?? url);
      if (target && path.endsWith('.pg')) {
        const homePath = path.replace(/^\/home\//, '');
        pageSources[homePath] = { url: source ?? target, selector };
      }
      if (text) textPaths.add(path);
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
