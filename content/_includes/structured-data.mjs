// schema.org JSON-LD for the homepage, so search engines and other crawlers
// can read who David is without parsing the layout: the person (roles,
// schools, awards, profiles) and his published papers. Built from the same
// content as the page; used through the `structuredData` filter.

export function structuredData({ site, cv, publications, skills }) {
  const person = `${site.url}/#person`;
  const school = (name) => ({ '@type': 'CollegeOrUniversity', name });
  const [current, ...previous] = cv.education;
  const [degrees, field] = current.title.split(' in ');

  const papers = publications
    .filter((paper) => !/^submitted to/i.test(paper.data.venue))
    .map((paper) => ({
      '@type': 'ScholarlyArticle',
      headline: paper.data.title,
      datePublished: String(paper.date.getUTCFullYear()),
      isPartOf: paper.data.venue,
      url: paper.data.links?.DOI ?? paper.data.url,
      author: paper.data.authors
        .split(', ')
        .map((name) =>
          name === site.authorName ? { '@id': person } : { '@type': 'Person', name },
        ),
    }));

  const graph = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'Person',
        '@id': person,
        name: site.name,
        url: site.url,
        image: `${site.url}${site.photo}`,
        email: `mailto:${site.email}`,
        telephone: site.phone,
        jobTitle: `${degrees.split(' & ')[0]} in ${field}`,
        description: site.description,
        affiliation: school(current.org),
        alumniOf: previous.map((item) => school(item.org)),
        address: {
          '@type': 'PostalAddress',
          addressLocality: site.location.split(', ')[0],
          addressRegion: site.location.split(', ')[1],
          addressCountry: 'US',
        },
        award: cv.awards.map((award) => award.title),
        knowsAbout: [
          'Distributed heterogeneous computing',
          'Processing-in-Memory',
          'Multi-GPU programming',
          'Task-based runtime systems',
          'Lossy compression',
          ...skills,
        ],
        sameAs: [site.github, site.linkedin, site.scholar, site.orcid],
      },
      { '@type': 'WebSite', url: site.url, name: site.name, author: { '@id': person } },
      ...papers,
    ],
  };
  // Safe inside <script>: no "</script>" can appear in the output.
  return JSON.stringify(graph).replaceAll('<', '\\u003c');
}
