// Build-time HTML → text extraction for the terminal filesystem.
//
// The terminal shows plain text for its `.pg` "page" entries. Rather than ship
// an HTML parser to every browser (and to the CLI) and re-parse on every read,
// we extract the text once here, at manifest-generation time, and commit the
// result. This is the single place HTML → text happens.
//
// linkedom is a dev/build dependency only — nothing here ships to the runtime.
// Using a real DOM keeps this output identical to what the browser's DOMParser
// produced when it did the same extraction at read time.

import { parseHTML } from 'linkedom';

const sectionText = (section) => {
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
};

export const extractPageText = (html, selector = '.holder') => {
  const { document } = parseHTML(html);
  const sections = [...document.querySelectorAll(selector)].map(sectionText).filter(Boolean);
  const text = sections.join('\n\n').replace(/\n{3,}/g, '\n\n');
  if (!text) throw new Error(`No readable content for selector ${selector}`);
  return text;
};
