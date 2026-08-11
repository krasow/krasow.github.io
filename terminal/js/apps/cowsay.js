(() => {
  'use strict';

  const wrap = (message, maximumWidth) => {
    const lines = [];
    let remaining = message;
    while (remaining.length > maximumWidth) {
      const breakpoint = remaining.lastIndexOf(' ', maximumWidth);
      const length = breakpoint > 0 ? breakpoint : maximumWidth;
      lines.push(remaining.slice(0, length));
      remaining = remaining.slice(length).trimStart();
    }
    lines.push(remaining);
    return lines;
  };

  const render = (message, maximumWidth = 60) => {
    const lines = wrap(message, Math.max(12, Math.min(60, maximumWidth)));
    const width = Math.max(...lines.map((line) => line.length));
    const bubble = lines.map((line, index) => {
      const first = index === 0;
      const last = index === lines.length - 1;
      const left = lines.length === 1 ? '<' : first ? '/' : last ? '\\' : '|';
      const right = lines.length === 1 ? '>' : first ? '\\' : last ? '/' : '|';
      return `${left} ${line.padEnd(width)} ${right}`;
    });

    return [
      ` ${'_'.repeat(width + 2)}`,
      ...bubble,
      ` ${'-'.repeat(width + 2)}`,
      '        \\   ^__^',
      '         \\  (oo)\\_______',
      '            (__)\\       )\\/\\',
      '                ||----w |',
      '                ||     ||',
    ].join('\n');
  };

  class CowsayApp {
    constructor(terminal) {
      this.terminal = terminal;
    }

    run(message = 'Moo.') {
      const style = getComputedStyle(this.terminal.ui.log);
      const fontSize = parseFloat(style.fontSize) || 14;
      const characterWidth = fontSize * 0.62;
      const maximumWidth = Math.floor(this.terminal.ui.log.clientWidth / characterWidth) - 6;
      this.terminal.write(render(message, maximumWidth), 'pth');
      return true;
    }
  }

  window.CowsayApp = CowsayApp;
})();
