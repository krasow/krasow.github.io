// Reading content/ from the .11ty.js templates that rebuild it in other
// shapes (the resume, the LinkedIn profile): the pieces of the homepage and
// CV they draw on, and a paper's status.

export const monthYear = (date) =>
  date.toLocaleString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });

export const profile = (collections) => ({
  cv: collections.cv[0].data,
  summary: collections.snippets[0].templateContent.trim(),
  skills: collections.all.find((page) => page.inputPath === './content/index.md').data.skills,
  experience: [...collections.experience].reverse(),
  publications: [...collections.publications].reverse(),
});

// Read from the venue: "under review" is under review, "Technical Report …"
// is a report, and anything else went through peer review.
export const paperStatus = (paper) => {
  if (/^under review/i.test(paper.data.venue)) return 'under review';
  if (/technical report/i.test(paper.data.venue)) return 'report';
  return 'reviewed';
};
