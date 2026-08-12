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
  const responseText = async (response, limit) => {
    const length = Number(response.headers?.get('content-length'));
    if (Number.isFinite(length) && length > limit) throw new Error('response is too large');
    if (!response.body?.getReader) {
      const text = await response.text();
      if (new TextEncoder().encode(text).byteLength > limit)
        throw new Error('response is too large');
      return text;
    }

    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) {
        await reader.cancel();
        throw new Error('response is too large');
      }
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) {
      bytes.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return new TextDecoder().decode(bytes);
  };
  const globRegex = (pattern) => {
    const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&');
    return new RegExp(`^${escaped.replace(/\*/g, '.*').replace(/\?/g, '.')}$`);
  };

  window.TerminalApp = { exact, maximum, readableError, responseText, globRegex };
})();
