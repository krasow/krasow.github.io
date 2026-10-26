// Builds the site from content/ into _site/. Run `npm run build` (or
// `npm run serve` to preview with live reload).
import markdownIt from 'markdown-it';
import markdownItAttrs from 'markdown-it-attrs';
import markdownItFootnote from 'markdown-it-footnote';
import { tex } from '@mdit/plugin-tex';
import site from './content/_data/site.json' with { type: 'json' };
import { generateManifest } from './terminal/generate-fs-manifest.mjs';

// Static files served as-is, at the same paths as their source.
const PASSTHROUGH = [
  '404.html',
  'CNAME',
  'assets',
  'common/css',
  'common/js',
  'common/gtag.js',
  'scripts',
  'terminal',
];

export default function (eleventyConfig) {
  const markdown = markdownIt({ html: true })
    .use(markdownItAttrs)
    .use(markdownItFootnote)
    // Leave math for MathJax to typeset in the browser.
    .use(tex, {
      delimiters: 'dollars',
      render: (math, displayMode) => {
        const escaped = markdown.utils.escapeHtml(math);
        return displayMode ? `<p>\\[${escaped}\\]</p>` : `\\(${escaped}\\)`;
      },
    });
  eleventyConfig.setLibrary('md', markdown);

  PASSTHROUGH.forEach((path) => eleventyConfig.addPassthroughCopy({ [path]: path }));
  eleventyConfig.addGlobalData('buildYear', new Date().getFullYear());

  // Inline Markdown for front matter fields (authors, venues, links).
  eleventyConfig.addFilter('md', (text) => (text ? markdown.renderInline(String(text)) : ''));
  // Rendered Markdown body as inline HTML: paragraphs become <br /><br /> breaks
  // so it fits inside the existing `.description` blocks.
  eleventyConfig.addFilter('unwrap', (html) =>
    html
      .trim()
      .replace(/^<p>|<\/p>$/g, '')
      .replace(/<\/p>\s*<p>/g, '<br /><br />'),
  );
  // Author list with the site owner's name wrapped in <strong class="author-me">.
  eleventyConfig.addFilter('authors', (authors) => {
    const escaped = markdown.utils.escapeHtml(authors);
    const me = markdown.utils.escapeHtml(site.authorName);
    return escaped.replaceAll(me, `<strong class="author-me">${me}</strong>`);
  });
  eleventyConfig.addFilter('year', (date) => date.getUTCFullYear());
  eleventyConfig.addFilter('month', (date) =>
    date.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' }),
  );
  eleventyConfig.addFilter('monthYear', (date) =>
    date.toLocaleString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' }),
  );
  eleventyConfig.addFilter('newestFirst', (items) => [...items].reverse());
  // Collection items whose front matter `key` equals `value` (or is set at all).
  eleventyConfig.addFilter('where', (items, key, value) =>
    items.filter((item) => (value === undefined ? item.data[key] : item.data[key] === value)),
  );
  // A talk's `links`, with a Poster button (after Slides) for its `poster:` id
  // from content/posters/. Unknown ids fail the build rather than drop the link.
  eleventyConfig.addFilter('talkLinks', (talk, posters) => {
    const links = Object.entries(talk.data.links ?? {});
    if (!talk.data.poster) return Object.fromEntries(links);
    const poster = posters.find((item) => item.data.id === talk.data.poster);
    if (!poster) throw new Error(`${talk.inputPath}: no poster with id ${talk.data.poster}`);
    links.splice(links.findIndex(([label]) => label === 'Slides') + 1, 0, ['Poster', poster.data.file]);
    return Object.fromEntries(links);
  });
  eleventyConfig.addFilter('json', (value) => JSON.stringify(value));

  // The terminal's filesystem manifest and page text are built from the
  // finished site, so they always match what's deployed.
  eleventyConfig.on('eleventy.after', ({ dir }) => generateManifest(dir.output));

  return {
    dir: { input: 'content', output: '_site' },
    templateFormats: ['md', 'njk'],
    // Content is plain Markdown; don't run it through a template language.
    markdownTemplateEngine: false,
    htmlTemplateEngine: 'njk',
  };
}
