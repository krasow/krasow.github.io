export default {
  tags: 'projects',
  title: 'Projects',
  layout: 'layouts/project.njk',
  eleventyComputed: {
    permalink: (data) => `/projects/${data.page.fileSlug}/`,
    title: (data) => data.card?.name ?? data.title,
  },
};
