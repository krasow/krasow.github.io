// The CLI's `chat` knowledge base (/terminal/data/chat.json). Answers are
// written here; facts (contact details, education, awards, papers, roles,
// skills) are pulled from content/ so the chat can't drift from the site.
// terminal/js/apps/chat.js validates the shape on load.

// Not shown elsewhere on the site, so it lives here; used by the graduation and
// hiring answers.
const EXPECTED_GRADUATION = 'December 2027';

const monthYear = (date) =>
  date.toLocaleString('en-US', { month: 'short', year: 'numeric', timeZone: 'UTC' });
const bullets = (lines) => lines.map((line) => `• ${line}`).join('\n');
const byId = (items, id) => {
  const item = items.find((entry) => entry.data.id === id);
  if (!item) throw new Error(`chat.11ty.js: no entry with id ${id}`);
  return item;
};
const bySlug = (items, slug) => {
  const item = items.find((entry) => entry.fileSlug === slug);
  if (!item) throw new Error(`chat.11ty.js: no entry named ${slug}`);
  return item;
};

export const data = {
  permalink: '/terminal/data/chat.json',
  eleventyExcludeFromCollections: true,
};

export function render({ site, collections }) {
  const cv = collections.cv[0].data;
  const skills = collections.all.find((page) => page.inputPath === './content/index.md').data
    .skills;
  const publications = [...collections.publications].reverse();
  const experience = [...collections.experience].reverse();
  const [, bachelors] = cv.education;
  const cunumericPaper = bySlug(publications, 'cunumeric-jl');
  const polymerPaper = bySlug(publications, 'polymerpim');
  const polymerPoster = byId(collections.posters, 'CSGF26');
  const nvidiaArticle = bySlug(collections.news, 'science-in-parallel');
  const summary = collections.snippets[0].templateContent.trim();
  const projects = collections.projects
    .filter((project) => project.data.card)
    .sort((a, b) => a.data.card.order - b.data.card.order);
  const firstAuthored = publications.filter((paper) =>
    paper.data.authors.startsWith(site.authorName),
  ).length;
  const skillWords = skills.map((skill) => skill.toLowerCase());
  const paperLink = (id) => {
    const paper = byId(publications, id);
    return { label: `${id} — ${paper.data.title}`, url: paper.data.links?.PDF ?? paper.data.url };
  };

  // prettier-ignore
  const entries = [
    {
      topic: 'cooked easter egg',
      keywords: [],
      phrases: ['am i cooked'],
      answer: 'Yes.',
    },
    {
      topic: 'name',
      keywords: ['name', 'named', 'called', 'full name'],
      phrases: ['what should i call'],
      answer: `His name is ${site.name}.`,
    },
    {
      topic: 'current role and university',
      keywords: ['who', 'identity', 'bio', 'biography', 'introduce', 'phd candidate', 'northwestern', 'student', 'current', 'school', 'university', 'advisor', 'peter dinda'],
      phrases: ['about david', 'tell me about david', 'quick bio'],
      answer: 'David is a Computer Science Ph.D. candidate at Northwestern University, advised by Peter Dinda.',
      hint: 'Try asking about his work in Julia, lossy compression, or PIM.',
      readMore: { command: 'cat /home/about.pg' },
    },
    {
      topic: 'summary',
      keywords: ['summary', 'overview', 'pitch', 'strengths', 'strength', 'yourself', 'fit'],
      phrases: ['why should we hire', 'why hire', 'good fit', 'tell me about yourself', 'elevator pitch', 'what makes him'],
      answer: summary,
      hint: 'Ask about his research, skills, or availability.',
      readMore: {
        prompt: 'Would you like to open his CV? (y/n)',
        command: 'open /home/cv.pg',
      },
    },
    {
      topic: 'Northwestern experience',
      keywords: ['peter dinda'],
      phrases: ['at northwestern', 'northwestern research', 'northwestern work', 'doing for phd', 'phd research', 'research is he doing', 'doctoral work'],
      answer: 'At Northwestern University, David is a Computer Science Ph.D. candidate advised by Peter Dinda. His work there includes cuNumeric.jl, for running Julia array programs across distributed GPU systems, and PolymerPIM, which uses deferred execution to make Processing-in-Memory systems both easy to program and fast.',
      hint: 'Try asking about cuNumeric.jl or PolymerPIM.',
    },
    {
      topic: 'Clemson experience',
      keywords: ['fthpc', 'jon calhoun'],
      phrases: ['at clemson', 'at argonne', 'clemson experience', 'clemson research', 'clemson work', 'undergraduate research', 'group undergraduate', 'cluster competition', 'student cluster'],
      answer: `At Clemson University, David earned a ${bachelors.title.replace(' · ', ' in ')} and conducted undergraduate research with the FTHPC group under Jon Calhoun. He collaborated with Argonne National Laboratory on statistical methods for predicting error-bounded lossy compression ratios for scientific data. He also competed in the Student Cluster Competition at SC '21 and IndySCC at SC '22, building a cluster with Dell and Intel.`,
      readMore: {
        prompt: 'Would you like to read more about the compression project? (y/n)',
        command: 'open /home/projects/compression',
      },
    },
    {
      topic: 'graduation',
      keywords: ['graduation'],
      phrases: ['when does he graduate', 'when will he graduate', 'when is he graduating', 'expect to graduate', 'expected graduation', 'graduation date', 'when does he finish', 'when will he finish', 'when is he done'],
      answer: `David expects to graduate from Northwestern University with his Ph.D. in Computer Science in ${EXPECTED_GRADUATION}.`,
      hint: 'Ask whether he is looking for a job or internship.',
    },
    {
      topic: 'education',
      keywords: ['education', 'school', 'go', 'attend', 'attended', 'graduate', 'graduated', 'study', 'studied', 'degree', 'degrees', 'major', 'undergrad', 'college', 'clemson', 'northwestern', 'bachelor', 'masters', 'doctorate', 'doctoral', 'computer engineering', 'computer science'],
      answer: `David's education:\n${bullets(cv.education.map((school) => `${school.title} — ${school.org} (${school.date})`))}`,
    },
    {
      topic: 'resume',
      keywords: ['resume', 'résumé'],
      answer: "David's resume is available as a PDF.",
      readMore: {
        prompt: 'Would you like to open his resume? (y/n)',
        command: 'open /home/resume.pdf',
      },
    },
    {
      topic: 'curriculum vitae',
      keywords: ['cv', 'curriculum vitae', 'vitae'],
      phrases: ['full cv'],
      answer: "David's full curriculum vitae is available on this site.",
      readMore: {
        prompt: 'Would you like to open his CV? (y/n)',
        command: 'open /home/cv.pg',
      },
    },
    {
      topic: 'programming background',
      keywords: ['programming', 'language', 'languages', 'skills', 'technical skills', 'technology', 'technologies', 'tools', 'compiler', 'compilers', 'cgra', 'cgras', 'fpga', 'fpgas', ...skillWords],
      answer: `David's site lists these skills: ${skills.join(', ')}. Beyond GPUs, he has also worked with CGRAs and FPGAs.`,
      readMore: {
        prompt: 'Would you like to open his CV? (y/n)',
        command: 'open /home/cv.pg',
      },
    },
    {
      topic: 'experience and background',
      keywords: ['experience', 'background', 'career', 'professionally', 'employment', 'employer', 'workplace', 'history'],
      phrases: ['where has he worked', 'where did he work', 'places he worked', 'where was he a researcher', 'national laboratories', 'both argonne and los alamos', 'internship before', 'previous internships', 'past internships', 'prior internships', 'industry experience', 'years of experience'],
      answer: `David began his research career at Clemson University and worked with Argonne National Laboratory on lossy compression for scientific data. He is now a Computer Science Ph.D. candidate at Northwestern University and collaborates with Los Alamos National Laboratory on distributed heterogeneous computing.\n\nRoles:\n${bullets(experience.map((job) => `${job.data.title}, ${job.data.org} (${monthYear(job.date)} – ${job.data.end})`))}`,
      readMore: { command: 'cat /home/experience.pg' },
    },
    {
      topic: 'hiring availability',
      keywords: ['hire', 'hiring', 'available', 'availability', 'recruit', 'recruiting', 'job', 'jobs', 'internship', 'intern', 'full time', 'full-time', 'role', 'roles', 'position', 'positions', 'opportunity', 'opportunities', 'start'],
      phrases: ['open to work', 'available to work', 'looking for work', 'looking for a job', 'when can he start', 'start date', 'what kind of roles', 'type of position'],
      answer: `David is looking for an internship for 2027. He expects to graduate in ${EXPECTED_GRADUATION} and plans to begin looking for full-time positions around that time.`,
      readMore: {
        prompt: 'Would you like to download his contact card? (y/n)',
        command: 'download /home/contact.vcf',
      },
    },
    {
      topic: 'logistics',
      keywords: ['relocate', 'relocation', 'relocating', 'move', 'remote', 'remotely', 'hybrid', 'onsite', 'visa', 'sponsorship', 'sponsor', 'citizen', 'citizenship', 'clearance', 'authorization', 'authorized', 'salary', 'compensation', 'pay', 'references', 'reference', 'gpa', 'grades', 'academia', 'industry', 'postdoc'],
      phrases: ['work authorization', 'security clearance', 'salary expectations', 'open to hybrid work', 'open to remote work', 'hybrid work', 'remote work', 'industry or academia', 'academia or industry', 'consider a postdoc'],
      answer: `David's site doesn't cover that. He's happy to discuss it directly at ${site.email}.`,
      readMore: {
        prompt: 'Would you like to see his contact info? (y/n)',
        command: 'cat /home/contact.md',
      },
    },
    {
      topic: 'scheduling research',
      keywords: ['scheduling', 'scheduler', 'schedulers', 'resource scheduling', 'task scheduling'],
      phrases: ['scheduling work', 'scheduling research'],
      answer: 'David studies scheduling across distributed heterogeneous systems. His broader systems work includes cuNumeric.jl for distributed Julia and GPU applications and PolymerPIM and Legion-based support for PIM accelerators.',
      readMore: {
        prompt: 'Would you like to see his related projects? (y/n)',
        command: 'ls /home/projects',
      },
    },
    {
      topic: 'projects',
      keywords: ['project', 'projects', 'portfolio', 'built'],
      phrases: ['what has he built', 'what is he working on'],
      answer: `David's projects:\n${bullets(projects.map((project) => `${project.data.card.name} — ${project.data.card.org}`))}`,
      command: 'ls /home/projects',
    },
    {
      topic: 'open source',
      keywords: ['open source', 'opensource', 'contribution', 'contributions', 'contributor', 'contributed'],
      phrases: ['open source projects', 'open source work'],
      answer: 'David develops cuNumeric.jl (JuliaLegate) and Legion PIM (PrescienceLab), and contributed to Libpressio, an Argonne compression library. His lossy compression research code is in the FTHPC Correlation_Compressibility repository.',
      links: [
        { label: 'cuNumeric.jl', url: 'https://github.com/JuliaLegate/cuNumeric.jl/' },
        { label: 'Legion PIM', url: 'https://github.com/PrescienceLab/upmem-legion' },
        { label: 'Libpressio', url: 'https://github.com/CODARcode/libpressio' },
        { label: 'Correlation_Compressibility', url: 'https://github.com/FTHPC/Correlation_Compressibility' },
        { label: 'GitHub profile', url: site.github },
      ],
    },
    {
      topic: 'teamwork and leadership',
      keywords: ['team', 'teams', 'teamwork', 'leadership', 'lead', 'led', 'collaborate', 'collaboration', 'collaborative', 'collaborator', 'collaborators'],
      phrases: ['worked on a team', 'leadership experience'],
      answer: `David has competed on Clemson's Student Cluster Competition teams (SC '21 and IndySCC at SC '22), collaborates across Northwestern, Argonne, and Los Alamos National Laboratory, and teaches and mentors students:\n${bullets(cv.teaching.map((item) => `${item.title} — ${item.org} (${item.date})`))}`,
    },
    {
      topic: 'AI and machine learning',
      keywords: ['machine learning', 'ml', 'ai', 'deep learning', 'pytorch', 'tensorflow', 'llm', 'llms', 'neural'],
      phrases: ['machine learning experience', 'ai experience'],
      answer: "David's site doesn't list machine learning work. His research asks how emerging AI-focused heterogeneous accelerators can support the workloads of HPC domain scientists.",
      hint: 'Ask about his research or skills.',
    },
    {
      topic: 'research interests',
      keywords: ['work', 'research', 'hpc', 'high performance computing', 'distributed', 'heterogeneous', 'accelerators', 'domain scientists', 'thesis', 'dissertation', 'passionate', 'passion', 'interested', 'interests', 'excited'],
      phrases: ['phd thesis', 'distributed systems', 'he passionate about', 'he excited about', 'he interested in', 'what interests him', 'what does he do', 'what does he study', 'area of cs', 'scientific workloads', 'emerging accelerators', 'intended users', 'accelerator research'],
      answer: 'David studies distributed heterogeneous computing, especially ways to make emerging accelerators practical for scientific workloads and the domain scientists who run them. His current work explores scheduling across distributed heterogeneous systems.',
    },
    {
      topic: 'compression GitHub repository',
      keywords: [],
      phrases: ['compression github', 'compression repo', 'compression repository', 'github compression work'],
      answer: "David's correlation-compressibility work is available in the FTHPC Correlation_Compressibility repository.",
      links: [{ label: 'GitHub', url: 'https://github.com/FTHPC/Correlation_Compressibility' }],
    },
    {
      topic: 'lossy compression research',
      keywords: ['compression', 'lossy', 'prediction', 'scientific data', 'argonne', 'clemson', 'compressibility'],
      phrases: ['lossy compression', 'compression work', 'compression papers', 'compressibility work', 'compression ratios'],
      answer: 'David began his research at Clemson in collaboration with Argonne National Laboratory, developing statistical methods to predict error-bounded lossy compression ratios for scientific data.\n\nRelated papers:',
      links: ['CLUSTER23', 'IJHPCA23', 'DRBSD21'].map(paperLink),
      readMore: { command: 'open /home/projects/compression' },
    },
    {
      topic: 'cuNumeric.jl GitHub repository',
      keywords: [],
      phrases: ['cunumeric github', 'cunumeric repo', 'cunumeric repository'],
      answer: 'The cuNumeric.jl repository is hosted by JuliaLegate on GitHub.',
      links: [{ label: 'GitHub', url: 'https://github.com/JuliaLegate/cuNumeric.jl/' }],
    },
    {
      topic: 'cuNumeric.jl',
      keywords: ['cunumeric', 'julia', 'gpu', 'gpus', 'arrays', 'ndarray', 'legate', 'constellation', 'nvidia', 'ipdps'],
      phrases: ['what is cunumeric', 'cunumeric relate', 'julia work', 'julia project', 'julia array project', 'gpu work'],
      answer: `cuNumeric.jl is a Julia array library for running numerical programs across distributed, heterogeneous systems with minimal code changes. Its NDArray type is designed as a drop-in alternative to Julia's standard Array, while the Legate ecosystem uses Legion underneath to organize deferred computations, manage data, and schedule work across resources such as NVIDIA GPUs.\n\nIt grew into an industry collaboration with NVIDIA, now in beta ("${nvidiaArticle.data.title}", ${nvidiaArticle.data.venue}, ${nvidiaArticle.date.getUTCFullYear()}).\n\nPaper: "${cunumericPaper.data.title}" (${cunumericPaper.data.venue}).`,
      links: [
        { label: 'GitHub', url: 'https://github.com/JuliaLegate/cuNumeric.jl/' },
        { label: 'Documentation', url: 'https://julialegate.github.io/cuNumeric.jl/dev/' },
        { label: nvidiaArticle.data.venue, url: nvidiaArticle.data.url },
      ],
      readMore: { command: 'open /home/projects/cunumeric' },
    },
    {
      topic: 'Legion runtime',
      keywords: ['legion', 'runtime'],
      phrases: ['what is legion', 'legion runtime', 'task based scheduler', 'data movement'],
      answer: "Legion is a task-based runtime for programming distributed, heterogeneous systems. It manages parallel work and data movement across different processors and accelerators. David's work connects to Legion in two ways: cuNumeric.jl uses the Legion-based Legate ecosystem to run array-oriented Julia applications across distributed systems and NVIDIA GPUs, while Legion PIM extends the runtime to support UPMEM Processing-in-Memory accelerators.",
      links: [
        { label: 'Legion', url: 'https://legion.stanford.edu/' },
        { label: 'cuNumeric.jl GitHub', url: 'https://github.com/JuliaLegate/cuNumeric.jl/' },
        { label: 'cuNumeric.jl documentation', url: 'https://julialegate.github.io/cuNumeric.jl/dev/' },
        { label: 'Legion PIM GitHub', url: 'https://github.com/PrescienceLab/upmem-legion' },
      ],
    },
    {
      topic: 'Legion PIM GitHub repository',
      keywords: [],
      phrases: ['legion pim github', 'legion pim repo', 'legion pim repository', 'legion pims github', 'legion pims repo', 'legion pims repository'],
      answer: 'The Legion PIM repository is hosted by PrescienceLab on GitHub.',
      links: [{ label: 'GitHub', url: 'https://github.com/PrescienceLab/upmem-legion' }],
    },
    {
      topic: 'PolymerPIM',
      keywords: ['polymerpim', 'polymer', 'fusion', 'deferred', 'asplos'],
      phrases: ['deferred execution', 'vertical and horizontal fusion', 'what is polymerpim'],
      answer: `PolymerPIM is David's work on making Processing-in-Memory (PIM) systems both easy to program and fast, using deferred execution to enable vertical and horizontal fusion of PIM operations.\n\nPaper: "${polymerPaper.data.title}" (${polymerPaper.data.venue}).\nPoster: "${polymerPoster.data.title}", ${polymerPoster.data.venue} (${monthYear(polymerPoster.date)}).`,
      links: [{ label: 'Poster', url: polymerPoster.data.file }],
    },
    {
      topic: 'processing in memory',
      keywords: ['pim', 'pims', 'processing in memory', 'upmem', 'memory accelerator', 'legion', 'legionpim'],
      phrases: ['what is pim', 'what is processing in memory', 'pim work', 'pim accelerators', 'legion pim', 'upmem legion'],
      answer: "Processing-in-Memory (PIM) places compute capabilities in or near memory so data can be processed without repeatedly moving it to a traditional CPU or GPU. This can reduce the cost and bottleneck of data movement for memory-intensive workloads. David's Legion PIM work integrates UPMEM's PIM processors into the Legion runtime, using Legion to schedule tasks and manage data transfers while making the accelerators easier to program. His newer work, PolymerPIM, uses deferred execution to fuse PIM operations.",
      hint: 'Try asking about PolymerPIM.',
      links: [{ label: 'Legion PIM GitHub', url: 'https://github.com/PrescienceLab/upmem-legion' }],
      readMore: { command: 'open /home/projects/legionpim' },
    },
    {
      topic: 'Los Alamos National Laboratory',
      keywords: ['lanl', 'los alamos', 'pat mccormick', 'laboratory', 'lab'],
      phrases: ['at lanl', 'lanl collaboration'],
      answer: 'David collaborates with Pat McCormick at Los Alamos National Laboratory on scheduling for distributed heterogeneous systems. In summer 2024 he was a visiting student there, adding UPMEM PIM support to the Legion runtime.',
    },
    {
      topic: 'Department of Energy fellowship',
      keywords: ['doe', 'csgf', 'fellow', 'fellowship', 'computational science graduate fellowship'],
      answer: 'David is a recipient of the U.S. Department of Energy Computational Science Graduate Fellowship (DOE CSGF).',
    },
    {
      topic: 'awards',
      keywords: ['award', 'awards', 'honor', 'honors', 'prize', 'won', 'win', 'competition', 'first place', 'best poster'],
      answer: `David's awards and fellowships:\n${bullets(cv.awards.map((award) => `${award.title} — ${award.org} (${award.date})`))}`,
    },
    {
      topic: 'teaching and mentoring',
      keywords: ['teaching', 'teach', 'taught', 'ta', 'teaching assistant', 'mentor', 'mentoring', 'mentored', 'course', 'class'],
      answer: `David's teaching and mentoring:\n${bullets(cv.teaching.map((item) => `${item.title} — ${item.org} (${item.date})`))}`,
    },
    {
      topic: 'phone number',
      keywords: ['phone', 'number', 'telephone', 'mobile', 'call'],
      answer: `David's phone number is ${site.phone}.`,
      readMore: {
        prompt: 'Would you like to see more contact info? (y/n)',
        command: 'cat /home/contact.md',
      },
    },
    {
      topic: 'physical address',
      keywords: ['address', 'mailing address', 'physical address', 'street address'],
      answer: `David is based in ${site.location}. A street address is not publicly listed.`,
    },
    {
      topic: 'email address',
      keywords: ['email', 'email address', 'mail'],
      answer: `David's email address is ${site.email}.`,
      readMore: {
        prompt: 'Would you like to see more contact info? (y/n)',
        command: 'cat /home/contact.md',
      },
    },
    {
      topic: 'contact',
      keywords: ['contact', 'card', 'vcard', 'vcf', 'download', 'save', 'reach', 'location', 'located', 'live', 'lives', 'based', 'chicago'],
      phrases: ['contact card', 'download contact', 'save contact', 'recruiter reach', 'recruiter contact'],
      answer: `David is based in ${site.location}. You can reach him at ${site.email} or run \`download contact.vcf\` to save his contact card.`,
      readMore: {
        prompt: 'Would you like to download his contact card? (y/n)',
        command: 'download /home/contact.vcf',
      },
    },
    {
      topic: 'meeting',
      keywords: ['meeting', 'meet', 'schedule', 'calendar', 'appointment', 'interview'],
      phrases: ['schedule a meeting', 'set up a call', 'book a time', 'set up a meeting', 'schedule a call'],
      answer: 'You can book a time with David on his calendar.',
      links: [{ label: 'Schedule a meeting', url: site.calendar }],
    },
    {
      topic: 'GitHub profile',
      keywords: ['github', 'github profile', 'github account', 'repositories', 'repos'],
      phrases: ['davids github', 'david github', 'his github', 'github profile', 'github account'],
      answer: "David's GitHub profile is @Krasow, where you can find his public repositories and contributions.",
      links: [{ label: 'GitHub', url: site.github }],
    },
    {
      topic: 'LinkedIn',
      keywords: ['linkedin'],
      answer: "David's LinkedIn profile:",
      links: [{ label: 'LinkedIn', url: site.linkedin }],
    },
    {
      topic: 'Google Scholar',
      keywords: ['scholar', 'google scholar', 'citations', 'citation'],
      answer: "David's Google Scholar profile:",
      links: [{ label: 'Google Scholar', url: site.scholar }],
    },
    {
      topic: 'news and updates',
      keywords: ['news', 'update', 'updates', 'announcement', 'announcements', 'latest', 'recent'],
      answer: "Here are David's latest updates:",
      command: 'cat /home/news.pg',
    },
    {
      topic: 'publications and papers',
      keywords: ['publication', 'publications', 'paper', 'papers', 'published', 'writing', 'list', 'submitted', 'review'],
      phrases: ['research papers', 'first author', 'first authored'],
      answer: `David's publications and technical reports:\n${bullets(publications.map((paper) => `${paper.data.title} — ${paper.data.venue} (${paper.date.getUTCFullYear()})`))}\n\nHe is first author on ${firstAuthored} of ${publications.length}. The published ones are in ~/publications:`,
      command: 'ls /home/publications',
    },
    {
      topic: 'presentations and talks',
      keywords: ['presentation', 'presentations', 'talk', 'talks', 'slide', 'slides', 'lecture', 'lectures'],
      phrases: ['research talks', 'conference slides'],
      answer: "Here are David's listed presentations:",
      command: 'ls /home/presentations',
    },
    {
      topic: 'posters',
      keywords: ['poster', 'posters', 'conference poster', 'research poster'],
      answer: "Here are David's listed posters:",
      command: 'ls /home/posters',
    },
  ];

  return `${JSON.stringify(
    {
      entries,
      fallback: `I don't know that from David's site. Try asking about his education, research, projects, skills, availability, publications, or contact information, or email him at ${site.email}.`,
    },
    null,
    2,
  )}\n`;
}
