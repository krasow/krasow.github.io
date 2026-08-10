(() => {
  'use strict';

  const KNOWLEDGE_URL = '/assets/documents/terminal/chat.json';
  const COMMON_WORDS = new Set([
    'about', 'can', 'david', 'did', 'do', 'does', 'from', 'has', 'have', 'he',
    'help', 'his', 'how', 'into', 'is', 'krasowska', 'me', 'the', 'their', 'them',
    'they', 'this', 'was', 'what', 'when', 'where', 'which', 'with', 'you', 'your',
  ]);

  const wordsIn = (text) => text.toLowerCase().match(/[a-z0-9]+/g) ?? [];

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
      const { entries, fallback, schools, vocabulary } = await this.load();
      const schoolAnswer = this.answerSchoolQuestion(question, schools);
      if (schoolAnswer) return { answer: schoolAnswer };

      const query = new Set(wordsIn(question).map((word) => this.correct(word, vocabulary)));
      const normalized = question.toLowerCase();
      const score = (entry) => entry.words.filter((word) => query.has(word)).length
        + (entry.phrases.some((phrase) => normalized.includes(phrase)) ? 10 : 0);
      const best = entries.reduce((a, b) => score(b) > score(a) ? b : a);
      return score(best) ? best : { answer: fallback };
    }

    async load() {
      if (!this.data) this.data = fetch(this.url, { cache: 'no-store' })
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json();
        })
        .then(({ entries, fallback, schools }) => {
          const indexed = entries.map((entry) => ({
            ...entry,
            words: entry.keywords.flatMap(wordsIn),
            phrases: entry.phrases ?? [],
          }));
          return {
            entries: indexed,
            fallback,
            schools,
            vocabulary: new Set(indexed.flatMap((entry) => entry.words)),
          };
        })
        .catch((error) => {
          this.data = null;
          throw error;
        });
      return this.data;
    }

    answerSchoolQuestion(question, schools) {
      const match = question.toLowerCase().match(/\b(?:go(?:es)? to|attend(?:s)?|study at)\s+([a-z]+)/);
      if (!match) return null;
      const school = [...schools.current, ...schools.former].reduce((best, candidate) => {
        const distance = editDistance(match[1], candidate);
        return distance < best.distance ? { name: candidate, distance } : best;
      }, { name: match[1], distance: 3 });
      const name = school.distance <= 2 ? school.name : match[1];
      if (schools.current.includes(name)) return schools.currentAnswer;
      if (schools.former.includes(name)) return schools.formerAnswer;
      return schools.otherAnswer;
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
