(() => {
  'use strict';

  const exact = (args, count, action) => {
    if (args.length !== count) return false;
    action();
    return true;
  };
  const maximum = (args, count, action) => {
    if (args.length > count) return false;
    action();
    return true;
  };
  const readableError = (error, fallback = 'unable to read file') =>
    error.message === 'EISDIR'
      ? 'is a directory'
      : error.message === 'ENOENT'
        ? 'no such text file'
        : fallback;
  const globRegex = (pattern) => {
    const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&');
    return new RegExp(`^${escaped.replace(/\*/g, '.*').replace(/\?/g, '.')}$`);
  };

  window.TerminalApp = { exact, maximum, readableError, globRegex };
})();
