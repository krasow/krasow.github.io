// /llms.txt (https://llmstxt.org): a plain-Markdown summary of the site for AI
// agents and other tools, built from the same content as the pages.

const monthYear = (date) =>
  date.toLocaleString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
const plain = (text) => String(text ?? '').replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

export const data = {
  permalink: '/llms.txt',
  eleventyExcludeFromCollections: true,
};

export function render({ site, collections }) {
  const absolute = (url) => (url.startsWith('/') ? `${site.url}${url}` : url);
  const link = (label, url) => (url ? `[${label}](${absolute(url)})` : label);
  const cv = collections.cv[0].data;
  const summary = collections.snippets[0].templateContent.trim();
  const skills = collections.all.find((page) => page.inputPath === './content/index.md').data
    .skills;
  const newest = (items) => [...items].reverse();
  const projects = collections.projects
    .filter((project) => project.data.card)
    .sort((a, b) => a.data.card.order - b.data.card.order);
  const section = (title, lines) => `## ${title}\n\n${lines.map((line) => `- ${line}`).join('\n')}`;

  return `${[
    `# ${site.name}`,
    `> ${site.description}`,
    `${summary}\n\nExpected graduation: ${site.expectedGraduation}.`,
    section('Contact', [
      `Email: ${site.email}`,
      `Phone: ${site.phone}`,
      `Location: ${site.location}`,
      link('Schedule a meeting', site.calendar),
      link('Resume (PDF)', site.resume),
      link('Contact card (vCard)', site.contactCard),
      `${link('GitHub', site.github)}, ${link('LinkedIn', site.linkedin)}, ${link('Google Scholar', site.scholar)}, ${link('ORCID', site.orcid)}`,
    ]),
    section(
      'Education',
      cv.education.map(
        (school) =>
          `${school.title}, ${school.org}, ${school.location} (${school.date})${school.desc ? `. ${school.desc}` : ''}`,
      ),
    ),
    section(
      'Experience',
      newest(collections.experience).map(
        (job) =>
          `**${job.data.title}**, ${job.data.org} (${monthYear(job.date)} – ${job.data.end}): ${plain(job.data.summary)}`,
      ),
    ),
    section(
      'Publications',
      newest(collections.publications).map((paper) => {
        const url = paper.data.links?.DOI ?? paper.data.url ?? paper.data.links?.Code;
        const venue = /^under review/i.test(paper.data.venue) ? paper.data.short : paper.data.venue;
        return `${link(paper.data.title, url)}. ${paper.data.authors}. ${venue}, ${paper.date.getUTCFullYear()}.`;
      }),
    ),
    section(
      'Talks',
      newest(collections.presentations).map(
        (talk) =>
          `${link(talk.data.title, talk.data.links?.Slides)}. ${talk.data.venue} (${monthYear(talk.date)}).`,
      ),
    ),
    section(
      'Posters',
      newest(collections.posters).map(
        (poster) =>
          `${link(poster.data.title, poster.data.file)}. ${poster.data.venue} (${monthYear(poster.date)}).`,
      ),
    ),
    section(
      'Awards',
      cv.awards.map((award) => `${award.title}, ${award.org} (${award.date})`),
    ),
    section(
      'Teaching and mentoring',
      cv.teaching.map((item) => `${item.title}, ${item.org} (${item.date})`),
    ),
    section(
      'Projects',
      projects.map(
        (project) => `${link(project.data.card.name, project.url)}: ${project.data.card.org}`,
      ),
    ),
    section('Skills', [skills.join(', ')]),
    section('Pages', [
      link('CV', '/pages/cv.html'),
      link('About', '/pages/about.html'),
      link('Presentations', '/pages/presentations.html'),
      link('News', '/pages/news.html'),
      link('Terminal version of the site', '/terminal/'),
    ]),
  ].join('\n\n')}\n`;
}
