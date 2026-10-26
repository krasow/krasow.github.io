// resume/linkedin.txt: the LinkedIn profile, field by field, built from the
// same content as the site so it can be pasted in (LinkedIn has no API for
// editing a profile). After content changes, its git diff shows which fields
// to update. Written by `npm run resume` along with the resume. The file holds
// only what gets pasted: papers under review are left out until accepted.

// LinkedIn's field limits; the build fails rather than writing text that
// LinkedIn would cut off.
const LIMITS = { headline: 220, about: 2600, title: 100, description: 2000 };

const monthYear = (date) =>
  date.toLocaleString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
const checked = (field, text, limit) => {
  if (text.length > limit)
    throw new Error(`linkedin.11ty.js: ${field} is ${text.length} characters (limit ${limit})`);
  return text;
};
const fields = (entries) =>
  entries
    .filter(([, value]) => value)
    .map(([name, value]) => (value.includes('\n') ? `${name}:\n${value}` : `${name}: ${value}`))
    .join('\n');
const section = (title, items) =>
  [`${'='.repeat(72)}\n${title.toUpperCase()}\n${'='.repeat(72)}`, ...items].join('\n\n');

export const data = {
  permalink: () => (process.env.BUILD_DOCUMENTS ? '/resume/linkedin.txt' : false),
  eleventyExcludeFromCollections: true,
};

export function render({ site, collections }) {
  const cv = collections.cv[0].data;
  const summary = collections.snippets[0].templateContent.trim();
  const skills = collections.all.find((page) => page.inputPath === './content/index.md').data
    .skills;
  const experience = [...collections.experience].reverse();
  const publications = [...collections.publications].reverse();
  const published = publications.filter((paper) => !/^submitted to/i.test(paper.data.venue));

  const about = `${summary}\n\nPublications, talks, and projects: ${site.url}`;

  return `${[
    section('Intro', [
      fields([
        ['Headline', checked('Headline', site.headline, LIMITS.headline)],
        ['Location', site.location],
        ['Website', site.url],
      ]),
    ]),
    section('About', [checked('About', about, LIMITS.about)]),
    section(
      'Experience',
      experience.map((job) =>
        fields([
          ['Title', checked(`${job.inputPath} title`, job.data.title, LIMITS.title)],
          ['Company or organization', job.data.org],
          ['Start date', monthYear(job.date)],
          ['End date', job.data.end],
          [
            'Description',
            checked(
              `${job.inputPath} description`,
              (job.data.resume ?? []).map((bullet) => `• ${bullet}`).join('\n'),
              LIMITS.description,
            ),
          ],
        ]),
      ),
    ),
    section(
      'Education',
      cv.education.map((school) => {
        const [degree, field] = school.title.split(' in ');
        const [start, end] = school.date.split(/\s*–\s*/);
        return fields([
          ['School', school.org],
          ['Degree', degree],
          ['Field of study', field],
          ['Start date', start],
          ['End date (or expected)', end === 'Now' ? `${site.expectedGraduation} (expected)` : end],
          ['Description', school.desc],
        ]);
      }),
    ),
    section(
      'Publications',
      published.map((paper) =>
        fields([
          ['Title', paper.data.title],
          ['Publication or publisher', paper.data.venue],
          ['Publication date', String(paper.date.getUTCFullYear())],
          ['Authors', paper.data.authors],
          ['Publication URL', paper.data.links?.DOI ?? paper.data.url],
        ]),
      ),
    ),
    section(
      'Honors & awards',
      cv.awards.map((award) =>
        fields([
          ['Title', checked(`award ${award.title}`, award.title, LIMITS.title)],
          ['Issuer', award.org],
          ['Issue date', String(award.date)],
        ]),
      ),
    ),
    section('Skills', [skills.join('\n')]),
  ].join('\n\n')}\n`;
}
