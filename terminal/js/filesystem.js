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
        return new Set(Array.isArray(paths)
          ? paths.filter((path) => typeof path === 'string' && path.startsWith('/'))
          : []);
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
      return [...this.paths].some((removed) => (
        path === removed || path.startsWith(`${removed}/`)
      ));
    }

    exists(path) {
      if (this.contains(path)) return false;
      if (this.directories[path] || this.fileRoutes.has(path) || this.hiddenFiles[path]) return true;
      const separator = path.lastIndexOf('/');
      const parent = path.slice(0, separator) || '/';
      const name = path.slice(separator + 1);
      return (this.directories[parent] ?? [])
        .some((entry) => entry.replace(/\/$/, '') === name);
    }

    remove(path, { recursive, currentDirectory }) {
      if (path !== '/home' && !path.startsWith('/home/')) return 'permission denied';
      if (!this.exists(path)) return 'no such file or directory';
      if (path === currentDirectory || currentDirectory.startsWith(`${path}/`)) {
        return 'resource busy';
      }
      if (this.directories[path] && !recursive) return 'is a directory';
      this.paths.add(path);
      return null;
    }
  }

  const addEntry = (directories, directory, entry) => {
    directories[directory] ??= [];
    if (!directories[directory].includes(entry)) directories[directory].push(entry);
  };

  const hydrate = (files, { directories, fileRoutes, textPaths }) => {
    files.forEach(({ path, url, text = true }) => {
      const parts = path.split('/').filter(Boolean);
      let directory = '/';
      parts.slice(0, -1).forEach((part) => {
        addEntry(directories, directory, `${part}/`);
        directory = `${directory === '/' ? '' : directory}/${part}`;
        directories[directory] ??= [];
      });

      addEntry(directories, directory, parts.at(-1));
      fileRoutes.set(path, url);
      if (text) textPaths.add(path);
    });
  };

  const loadManifest = async (url, fileSystem) => {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { files } = await response.json();
    if (!Array.isArray(files)) throw new Error('Invalid filesystem manifest');
    hydrate(files, fileSystem);
  };

  window.KrasowTerminalFileSystem = { VirtualTrash, hydrate, loadManifest };
})();
