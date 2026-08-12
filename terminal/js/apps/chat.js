(() => {
  'use strict';

  const KNOWLEDGE_URL = '/terminal/data/chat.json';
  const MAX_KNOWLEDGE_BYTES = 512 * 1024;
  const DATA_COMMAND = /^(?:cat|download|ls|open) \/[A-Za-z0-9._/-]+$/;
  const SIDE_EFFECTING_COMMANDS = new Set(['download', 'open']);
  const validDataCommand = (command) => {
    if (typeof command !== 'string' || !DATA_COMMAND.test(command)) return false;
    const path = command.slice(command.indexOf(' ') + 1);
    return path
      .split('/')
      .every((part, index) => index === 0 || (part && part !== '.' && part !== '..'));
  };
  const validLink = (value) => {
    if (typeof value !== 'string' || value.includes('\\') || /[\u0000-\u001f\u007f]/.test(value))
      return false;
    if (value.startsWith('/') && !value.startsWith('//')) return true;
    try {
      const url = new URL(value);
      return url.protocol === 'https:' && !url.username && !url.password;
    } catch {
      return false;
    }
  };
  const validEntry = (entry) =>
    entry &&
    typeof entry === 'object' &&
    !Array.isArray(entry) &&
    typeof entry.answer === 'string' &&
    Array.isArray(entry.keywords) &&
    entry.keywords.every((word) => typeof word === 'string') &&
    (entry.phrases === undefined ||
      (Array.isArray(entry.phrases) &&
        entry.phrases.every((phrase) => typeof phrase === 'string'))) &&
    (entry.hint === undefined || typeof entry.hint === 'string') &&
    (entry.command === undefined || validDataCommand(entry.command)) &&
    (entry.readMore === undefined ||
      (entry.readMore &&
        typeof entry.readMore === 'object' &&
        validDataCommand(entry.readMore.command) &&
        (entry.readMore.prompt === undefined || typeof entry.readMore.prompt === 'string'))) &&
    !(entry.command && entry.readMore) &&
    (entry.links === undefined ||
      (Array.isArray(entry.links) &&
        entry.links.every(
          (link) =>
            link &&
            typeof link === 'object' &&
            typeof link.label === 'string' &&
            validLink(link.url),
        )));
  // prettier-ignore
  const COMMON_WORDS = new Set([
    'a', 'about', 'an', 'and', 'area', 'at', 'came', 'can', 'david', 'did', 'do', 'does',
    'find', 'for', 'from', 'give', 'given', 'has', 'have', 'he', 'help', 'his', 'how', 'in',
    'into', 'is', 'kind', 'krasowska', 'me', 'tell', 'the', 'their', 'them', 'of', 'on',
    'they', 'this', 'to', 'was', 'what', 'when', 'where', 'which', 'with', 'you', 'your',
  ]);

  const wordsIn = (text) => text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  const normalizePhrase = (text) => wordsIn(text).join('');
  const normalizeReference = (word) => (['he', 'him'].includes(word) ? 'david' : word);

  const editDistance = (a, b) => {
    let row = [...Array(b.length + 1).keys()];
    for (let i = 1; i <= a.length; i += 1) {
      const next = [i];
      for (let j = 1; j <= b.length; j += 1) {
        next[j] = Math.min(
          next[j - 1] + 1,
          row[j] + 1,
          row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
        );
      }
      row = next;
    }
    return row[b.length];
  };

  class LocalChat {
    constructor(url = KNOWLEDGE_URL) {
      this.url = url;
      this.data = null;
    }

    async ask(question) {
      const { entries, fallback, vocabulary } = await this.load();
      const query = new Set(
        wordsIn(question)
          .map(normalizeReference)
          .map((word) => this.correct(word, vocabulary)),
      );
      const normalized = normalizePhrase(question);
      const phraseScore = (entry) =>
        entry.phrases.reduce((best, phrase) => {
          const matches =
            normalized.includes(phrase.compact) || phrase.words.every((word) => query.has(word));
          return matches ? Math.max(best, 10 + 2 * phrase.words.length) : best;
        }, 0);
      const score = (entry) =>
        entry.words.filter((word) => query.has(word)).length + phraseScore(entry);
      const best = entries.reduce((a, b) => (score(b) > score(a) ? b : a));
      return score(best) ? best : { answer: fallback };
    }

    async load() {
      if (!this.data)
        this.data = fetch(this.url, { cache: 'no-store' })
          .then(async (response) => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const text = await window.TerminalApp.responseText(response, MAX_KNOWLEDGE_BYTES);
            return JSON.parse(text);
          })
          .then(({ entries, fallback }) => {
            if (
              !Array.isArray(entries) ||
              !entries.length ||
              entries.length > 1_000 ||
              !entries.every(validEntry) ||
              typeof fallback !== 'string'
            )
              throw new Error('invalid knowledge data');
            const indexed = entries.map((entry) => ({
              ...entry,
              words: [
                ...new Set(
                  entry.keywords.flatMap(wordsIn).filter((word) => !COMMON_WORDS.has(word)),
                ),
              ],
              phrases: (entry.phrases ?? []).map((phrase) => ({
                compact: normalizePhrase(phrase),
                words: wordsIn(phrase).map(normalizeReference),
              })),
            }));
            return {
              entries: indexed,
              fallback,
              vocabulary: new Set(
                indexed.flatMap((entry) => [
                  ...entry.words,
                  ...entry.phrases.flatMap((phrase) => phrase.words),
                ]),
              ),
            };
          })
          .catch((error) => {
            this.data = null;
            throw error;
          });
      return this.data;
    }

    correct(word, vocabulary) {
      if (word.length < 4 || vocabulary.has(word) || COMMON_WORDS.has(word)) return word;
      const closest = [...vocabulary].reduce(
        (best, candidate) => {
          if (candidate.length < 4) return best;
          const distance = editDistance(word, candidate);
          return distance < best.distance ? { word: candidate, distance } : best;
        },
        { word, distance: 3 },
      );
      return closest.distance <= 1 ? closest.word : word;
    }
  }

  class ChatApp {
    constructor(terminal) {
      this.terminal = terminal;
      this.model = new LocalChat();
      this.session = false;
      this.mode = false;
      this.confirmation = null;
    }

    run(args) {
      if (args.length) this.ask(args.join(' '));
      else this.enter();
      return true;
    }

    async ask(question) {
      const thinking = document.createElement('p');
      thinking.className = 'ln hint';
      thinking.textContent = 'local model: thinking…';
      this.terminal.append(thinking);

      try {
        const result = await this.model.ask(question);
        thinking.remove();
        this.terminal.write(result.answer, 'pth');
        if (result.hint) this.terminal.write(result.hint, 'hint');
        result.links?.forEach(({ label, url }) => {
          const text = url.startsWith('/') ? `→ ${label}` : `→ ${label}: ${url}`;
          this.terminal.writeLink(url, text);
        });
        if (result.command) this.dispatchDataCommand(result.command);
        if (this.mode && result.readMore) {
          this.confirmation = result.readMore;
          this.updatePrompt();
          this.terminal.write(
            result.readMore.prompt ?? 'Would you like to read more? (y/n)',
            'hint',
          );
        }
      } catch (error) {
        thinking.remove();
        this.terminal.write('chat: the local knowledge model could not be loaded', 'err');
      }
    }

    enter() {
      this.session = true;
      this.mode = true;
      this.updatePrompt();
      this.terminal.write('Ask me about David. Type `exit` to return to the terminal.', 'hint');
    }

    leave() {
      this.session = false;
      this.mode = false;
      this.confirmation = null;
      this.updatePrompt();
      this.terminal.write('leaving chat', 'hint');
    }

    handle(command) {
      const normalized = command.toLowerCase();
      if (this.confirm(normalized)) return;
      if (['exit', 'quit'].includes(normalized)) this.leave();
      else if (['help', '?'].includes(normalized)) this.showHelp();
      else if (['hello', 'hi', 'hey'].includes(normalized)) {
        this.terminal.write(
          'Hello! Ask me anything about David, or type `help` for examples.',
          'pth',
        );
      } else if (command === ']') this.toggle();
      else this.ask(command);
    }

    confirm(answer) {
      if (!this.confirmation) return false;
      if (!['y', 'yes', 'n', 'no'].includes(answer)) {
        this.terminal.write('Please answer y or n.', 'hint');
        return true;
      }
      const command = this.confirmation.command;
      this.confirmation = null;
      this.updatePrompt();
      if (['y', 'yes'].includes(answer)) this.runTerminalCommand(command);
      return true;
    }

    handleBracket() {
      if (!this.session) return false;
      if (this.confirmation) this.terminal.write('Please answer y or n.', 'hint');
      else this.toggle();
      return true;
    }

    cancel() {
      if (this.confirmation) {
        this.confirmation = null;
        this.updatePrompt();
        this.terminal.write('Cancelled.', 'hint');
      } else if (this.mode) this.leave();
    }

    toggle() {
      this.mode = !this.mode;
      this.updatePrompt();
    }

    prompt(shellPrompt) {
      if (this.confirmation) return 'david:[y/n]>';
      return this.mode ? 'david:chat>' : shellPrompt;
    }

    updatePrompt() {
      this.terminal.ui.prompt.textContent = this.terminal.promptText();
    }

    runTerminalCommand(command) {
      if (!validDataCommand(command)) throw new Error('invalid knowledge command');
      this.terminal.echo(command, 'david:~$');
      const [name, ...args] = command.split(/\s+/);
      this.terminal.commands.get(name)?.(args);
    }

    dispatchDataCommand(command) {
      const name = command.split(/\s+/, 1)[0];
      if (!SIDE_EFFECTING_COMMANDS.has(name)) return this.runTerminalCommand(command);
      if (!this.mode) {
        this.terminal.write(`run \`${command}\` to continue`, 'hint');
        return;
      }
      this.confirmation = { command, prompt: `Run \`${command}\`? (y/n)` };
      this.updatePrompt();
      this.terminal.write(this.confirmation.prompt, 'hint');
    }

    showHelp() {
      this.terminal.write(
        [
          'Ask about David using natural language. For example:',
          '  what does he work on?',
          '  where did he study?',
          '  who is his advisor?',
          '  what publications does he have?',
          '  what is his work on PIM?',
          '  how can I contact him?',
          '',
          'Commands: help · ] toggle shell mode · exit · quit · Ctrl+C',
        ].join('\n'),
        'pth',
      );
    }
  }

  window.ChatApp = ChatApp;
})();
