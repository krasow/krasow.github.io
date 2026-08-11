(() => {
  'use strict';
  class HelpApp {
    constructor(terminal, sections) {
      this.terminal = terminal;
      this.sections = sections;
    }
    run(args) {
      return window.TerminalApp.exact(args, 0, () => this.show());
    }
    show(track = true) {
      const help = document.createElement('div');
      help.className = 'ln help';
      this.sections.forEach(([heading, rows]) => {
        const section = document.createElement('span');
        section.className = 'help-section';
        section.textContent = heading;
        help.append(section);
        rows.forEach(([command, description]) => {
          const commandNode = document.createElement('span');
          commandNode.className = 'help-command';
          commandNode.textContent = command;
          const descriptionNode = document.createElement('span');
          descriptionNode.className = 'help-description';
          descriptionNode.textContent = description;
          help.append(commandNode, descriptionNode);
        });
      });
      const note = document.createElement('span');
      note.className = 'help-note';
      note.textContent = 'Type any item shown by ls to open it.';
      help.append(note);
      this.terminal.append(help, track ? { type: 'help' } : null);
    }
  }
  window.HelpApp = HelpApp;
})();
