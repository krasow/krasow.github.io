(() => {
  'use strict';

  const KNOWLEDGE_URL = '/assets/documents/terminal/chat.json';
  const COMMON_WORDS = new Set([
    'about', 'area', 'came', 'can', 'david', 'did', 'do', 'does', 'find', 'from',
    'give', 'given', 'has', 'have', 'he', 'help', 'his', 'how', 'into', 'is', 'kind',
    'krasowska', 'me', 'tell', 'the', 'their', 'them',
    'they', 'this', 'was', 'what', 'when', 'where', 'which', 'with', 'you', 'your',
  ]);

  const wordsIn = (text) => text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  const normalizePhrase = (text) => wordsIn(text).join('');
  const normalizeReference = (word) => ['he', 'him'].includes(word) ? 'david' : word;

  const editDistance = (a, b) => {
    let row = [...Array(b.length + 1).keys()];
    for (let i = 1; i <= a.length; i += 1) {
      const next = [i];
      for (let j = 1; j <= b.length; j += 1) {
        next[j] = Math.min(next[j - 1] + 1, row[j] + 1,
          row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
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
      const query = new Set(wordsIn(question)
        .map(normalizeReference)
        .map((word) => this.correct(word, vocabulary)));
      const normalized = normalizePhrase(question);
      const score = (entry) => entry.words.filter((word) => query.has(word)).length
        + (entry.phrases.some((phrase) => normalized.includes(phrase.compact)
          || phrase.words.every((word) => query.has(word))) ? 10 : 0);
      const best = entries.reduce((a, b) => score(b) > score(a) ? b : a);
      return score(best) ? best : { answer: fallback };
    }

    async load() {
      if (!this.data) this.data = fetch(this.url, { cache: 'no-store' })
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json();
        })
        .then(({ entries, fallback }) => {
          const indexed = entries.map((entry) => ({
            ...entry,
            words: [...new Set(entry.keywords.flatMap(wordsIn))],
            phrases: (entry.phrases ?? []).map((phrase) => ({
              compact: normalizePhrase(phrase),
              words: wordsIn(phrase).map(normalizeReference),
            })),
          }));
          return {
            entries: indexed,
            fallback,
            vocabulary: new Set(indexed.flatMap((entry) => [
              ...entry.words,
              ...entry.phrases.flatMap((phrase) => phrase.words),
            ])),
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
      const closest = [...vocabulary].reduce((best, candidate) => {
        if (candidate.length < 4) return best;
        const distance = editDistance(word, candidate);
        return distance < best.distance ? { word: candidate, distance } : best;
      }, { word, distance: 3 });
      return closest.distance <= 2 ? closest.word : word;
    }
  }

  window.KrasowChat = { LocalChat };
})();
