// Checks which answer the CLI `chat` gives for questions recruiters and
// visitors are likely to ask. Runs the real matcher (terminal/js/apps/chat.js)
// against the built knowledge base (_site/terminal/data/chat.json), so run
// `npm test`, which builds first.
//
// When you add or reword chat answers in content/terminal/chat.11ty.js, add the
// questions you expect them to catch here.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const FALLBACK = 'fallback';

const knowledgeUrl = new URL('../_site/terminal/data/chat.json', import.meta.url);
const chatUrl = new URL('../terminal/js/apps/chat.js', import.meta.url);

const loadChat = () => {
  const knowledge = fs.readFileSync(knowledgeUrl, 'utf8');
  const window = { TerminalApp: { responseText: async (response) => response.text() } };
  const context = vm.createContext({
    window,
    fetch: async () => new Response(knowledge),
    Response,
    URL,
  });
  vm.runInContext(fs.readFileSync(chatUrl, 'utf8'), context);
  const topics = new Map(JSON.parse(knowledge).entries.map((entry) => [entry.answer, entry.topic]));
  const chat = new window.ChatApp({}).model;
  // Several answers (for "x and y" questions) are joined with " + ".
  return async (question) =>
    (await chat.askAll(question))
      .map((result) => topics.get(result.answer) ?? FALLBACK)
      .join(' + ');
};

// [question, topic of the answer it should get, or "a + b" when it should get several]
const EXPECTED = [
  // availability and timing
  ['is he available for an internship', 'hiring availability'],
  ['when can he start', 'hiring availability'],
  ['when is he available', 'hiring availability'],
  ['is he available for summer 2027', 'hiring availability'],
  ['is he looking for full time roles', 'hiring availability'],
  ['what kind of roles is he looking for', 'hiring availability'],
  ['what type of position does he want', 'hiring availability'],
  ['is he interested in industry or academia', 'logistics'],
  ['would he consider a postdoc', 'logistics'],
  ['how long of an internship can he do', 'hiring availability'],
  ['when does he graduate', 'graduation'],
  // location and logistics
  ['where is he located', 'contact'],
  ['is he willing to relocate', 'logistics'],
  ['can he work remotely', 'logistics'],
  ['is he open to hybrid work', 'logistics'],
  ['does he need visa sponsorship', 'logistics'],
  ['is he a us citizen', 'logistics'],
  ['does he have a security clearance', 'logistics'],
  ['what is his work authorization', 'logistics'],
  ['what are his salary expectations', 'logistics'],
  // skills
  ['what programming languages does he know', 'programming background'],
  ['does he know c++', 'programming background'],
  ['does he know python', 'programming background'],
  ['does he have cuda experience', 'programming background'],
  ['does he know mpi', 'programming background'],
  ['has he used llvm', 'programming background'],
  ['does he know rust', 'unlisted technologies'],
  ['does he have machine learning experience', 'AI and machine learning'],
  ['does he know pytorch', 'AI and machine learning'],
  ['does he have gpu programming experience', 'programming background'],
  ['does he have distributed systems experience', 'research interests'],
  ['does he know compilers', 'programming background'],
  ['has he worked with fpgas', 'programming background'],
  ['what are his strongest skills', 'programming background'],
  // experience
  ['how many years of experience does he have', 'experience and background'],
  ['has he done an internship before', 'experience and background'],
  ['does he have industry experience', 'experience and background'],
  ['where has he worked', 'experience and background'],
  ['what did he do at los alamos', 'Los Alamos National Laboratory'],
  ['what did he do at argonne', 'Clemson experience'],
  ['has he worked at nvidia', 'cuNumeric.jl'],
  ['what is his research about', 'research interests'],
  ['summarize his research', 'research interests'],
  ['what projects has he worked on', 'projects'],
  ['tell me about his phd thesis', 'research interests'],
  ['what open source projects has he contributed to', 'open source'],
  ['has he worked on a team', 'teamwork and leadership'],
  ['does he have leadership experience', 'teamwork and leadership'],
  ['has he mentored anyone', 'teaching and mentoring'],
  ['has he taught', 'teaching and mentoring'],
  // credentials
  ['what is his degree', 'education'],
  ['what is his gpa', 'logistics'],
  ['where did he go to school', 'education'],
  ['who is his advisor', 'current role and university'],
  ['what awards has he received', 'awards'],
  ['what fellowships does he have', 'Department of Energy fellowship'],
  ['what has he published', 'publications and papers'],
  ['how many papers does he have', 'publications and papers'],
  ['does he have any first author papers', 'publications and papers'],
  ['what talks has he given', 'presentations and talks'],
  // contact
  ['how do i contact him', 'contact'],
  ['what is his email', 'email address'],
  ['what is his phone number', 'phone number'],
  ['can i schedule a meeting with him', 'meeting'],
  ['does he have a linkedin', 'LinkedIn'],
  ['what is his google scholar', 'Google Scholar'],
  ['where is his github', 'GitHub profile'],
  ['can i see his resume', 'resume'],
  ['can i get his cv', 'curriculum vitae'],
  ['how can i get references', 'logistics'],
  // why him
  ['why should we hire him', 'summary'],
  ['what makes him a good fit', 'summary'],
  ['what are his strengths', 'summary'],
  ['what is he passionate about', 'research interests'],
  ['tell me about yourself', 'summary'],
  ['who is david', 'current role and university'],
  // motivation and side projects
  ['why does he work on pim', 'research motivation'],
  ['what motivates his research', 'research motivation'],
  ['what problem is he trying to solve', 'research motivation'],
  ['why does he care about heterogeneous computing', 'research motivation'],
  ['what is the constellation project', 'Constellation Project'],
  ['has he worked with databases', 'Constellation Project'],
  ['what is village', 'Village'],
  ['what can cunumeric do', 'cuNumeric.jl features'],
  ['does cunumeric support stencils', 'cuNumeric.jl features'],
  ['what is the cunumeric api like', 'cuNumeric.jl features'],
  ['what was his juliacon talk about', 'JuliaCon talk'],
  ['what did he win at sc22', 'ACM Student Research Competition'],
  ['what was his best poster', 'ACM Student Research Competition'],
  ['any advice on grad school', 'grad school advice'],
  ['did he give a talk about grad school', 'grad school advice'],
  ['what is he doing right now', 'Northwestern experience'],
  ['what year is he in', 'graduation'],
  ['what does he specialize in', 'research interests'],
  ['what is his expertise', 'research interests'],
  ['does he have software engineering experience', 'programming background'],
  ['does he use git', 'unlisted technologies'],
  ['does he know java', 'unlisted technologies'],
  ['has he done profiling', 'unlisted technologies'],
  ['what gpus has he used', 'hardware'],
  ['who are his coauthors', 'collaborators and co-authors'],
  ['who does he collaborate with', 'collaborators and co-authors'],
  ['what department is he in', 'current role and university'],
  ['what conferences has he attended', 'presentations and talks'],
  ['what is his most cited paper', 'Google Scholar'],
  ['what is his first paper', 'first paper'],
  ['what is his latest paper', 'latest papers'],
  ['what is his most recent publication', 'latest papers'],
  ['what was his last paper', 'latest papers'],
  ['newest publication', 'latest papers'],
  ['what is his most recent accepted publication', 'latest accepted papers'],
  ['what is his latest published paper', 'latest accepted papers'],
  ['what are his most recent accepted papers', 'latest accepted papers'],
  ['latest peer reviewed publication', 'latest accepted papers'],
  ['is the cunumeric paper accepted', 'latest papers'],
  ['what is ijhpca', 'lossy compression research'],
  ['what is the cluster paper about', 'lossy compression research'],
  ['can i work with him', 'working with David'],
  ['is he taking students', 'working with David'],
  ['how did he get into research', 'Clemson experience'],
  ['where did he start research', 'Clemson experience'],
  ['when did he start doing research', 'Clemson experience'],
  ['how did his research career begin', 'Clemson experience'],
  ['what time zone is he in', 'contact'],
  ['what are his pronouns', 'personal'],
  ['does he have a blog', 'social media'],
  ['is he looking for internships', 'hiring availability'],
  ['who was his undergrad advisor', 'Clemson experience'],
  ['who has he mentored', 'teaching and mentoring'],
  ['how long was he at argonne', 'experience and background'],
  ['has he worked in industry', 'experience and background'],
  ['what is his most recent talk', 'latest talk'],
  ['what is his newest poster', 'posters'],
  ['is he good at c++', 'programming background'],
  ['is he a good candidate', 'summary'],
  ['does he have orcid', 'ORCID'],
  ['what is his orcid id', 'ORCID'],
  ['what was his best poster', 'ACM Student Research Competition'],
  ['tell me more', 'what to ask'],
  ['what does he do with wonchan lee', 'co-author W. Lee'],
  ['has he worked with crotty', 'co-author A. Crotty'],
  ['who is robert underwood', 'co-author R. Underwood'],
  ['what does he do with nvidia', 'cuNumeric.jl'],
  // the chat and terminal themselves
  ['are you an ai', 'about this chat'],
  ['is this chatgpt', 'about this chat'],
  ['who are you', 'about this chat'],
  ['what commands can i use', 'terminal help'],
  ['how do i exit', 'terminal help'],
  ['is there a game', 'terminal help'],
  // regression: earlier questions must still route correctly
  ['when does david graduate', 'graduation'],
  ['is david looking for a job', 'hiring availability'],
  ['what awards has he won', 'awards'],
  ['does he teach', 'teaching and mentoring'],
  ['what is polymerpim', 'PolymerPIM'],
  ['what is pim', 'processing in memory'],
  ['what is legion', 'Legion runtime'],
  ['what is cunumeric', 'cuNumeric.jl'],
  ['tell me about the compression work', 'lossy compression research'],
  ['where does he live', 'contact'],
  ['what are his publications', 'publications and papers'],
  ['what posters does he have', 'posters'],
  ['what is the latest news', 'news and updates'],
  ['am i cooked', 'cooked easter egg'],
  ['what is his name', 'name'],
  ['what is he interested in', 'research interests'],
  ['tell me about david', 'current role and university'],
  // technical depth
  ['does he know openmp', 'unlisted technologies'],
  ['does he have experience with slurm', 'experience and background'],
  ['has he used supercomputers', 'hardware'],
  ['what hardware has he worked with', 'hardware'],
  ['does he do performance optimization', 'performance results'],
  ['has he done profiling or benchmarking', 'performance results'],
  ['does he know parallel programming', 'programming background'],
  ['does he have linux experience', 'experience and background'],
  ['does he know kubernetes', 'unlisted technologies'],
  ['does he have cloud experience', 'unlisted technologies'],
  ['how fast is cunumeric', 'performance results'],
  ['how does cunumeric compare to cupynumeric', 'performance results'],
  ['what is legate', 'cuNumeric.jl'],
  ['what is upmem', 'processing in memory'],
  ['what is a cgra', 'programming background'],
  // current work
  ['what is he working on now', 'Northwestern experience'],
  ['what is his most recent work', 'Northwestern experience'],
  ['what is his current project', 'Northwestern experience'],
  ['what is his biggest accomplishment', 'awards'],
  ['what results has he gotten', 'performance results'],
  ['who funds his research', 'Department of Energy fellowship'],
  ['what lab is he in', 'Prescience Lab'],
  ['what is prescience lab', 'Prescience Lab'],
  ['what year of his phd is he in', 'graduation'],
  // personal
  ['where is he from', 'personal'],
  ['what does he do for fun', 'personal'],
  ['does he have hobbies', 'personal'],
  ['does he speak polish', 'personal'],
  ['what languages does he speak', 'personal'],
  ['how old is he', 'personal'],
  // reading and citing
  ['where can i read his papers', 'publications and papers'],
  ['can i get a pdf of his paper', 'publications and papers'],
  ['can i get the bibtex', 'publications and papers'],
  ['what conferences has he presented at', 'presentations and talks'],
  ['has he presented at supercomputing', 'presentations and talks'],
  ['did he present at juliacon', 'JuliaCon talk'],
  ['what is his h-index', 'Google Scholar'],
  // contact variants
  ['best way to reach him', 'contact'],
  ['can i text him', 'phone number'],
  ['does he have twitter', 'social media'],
  ['does he have a personal website', 'this site'],
  // meta and small talk
  ['hi', 'greeting'],
  ['hello', 'greeting'],
  ['thanks', 'thanks'],
  ['help', FALLBACK],
  ['what can i ask you', 'what to ask'],
  ['what can you do', 'what to ask'],
  ['who made this site', 'this site'],
  ['was this site made with ai', 'this site'],
  ['how was this site built', 'this site'],
  ['where is the source code for this site', 'this site'],
  // typos
  ['were did he study', 'education'],
  ['who is his phd advisr', 'current role and university'],
  ['what is his emial', 'email address'],
  ['can i see his resme', 'resume'],
  ['does he know pyton', 'programming background'],
  ['what are his publicatons', 'publications and papers'],
  // role fit
  ['would he be a good fit for an hpc engineer role', 'summary'],
  ['is he a good fit for a gpu performance engineer position', 'summary'],
  ['would he fit a compiler engineer role', 'programming background'],
  ['is he qualified for a research scientist position', 'summary'],
  ['would he be interested in ml infrastructure', 'hiring availability'],
  ['can he do systems programming', 'programming background'],
  ['does he have backend experience', 'experience and background'],
  ['is he a software engineer', 'programming background'],
  // company and lab fit
  ['would he want to work at nvidia', 'hiring availability'],
  ['has he worked with amd gpus', 'hardware'],
  ['does he have experience with intel', 'experience and background'],
  ['is he interested in national labs', 'experience and background'],
  ['has he interned at a national lab', 'experience and background'],
  ['has he worked with doe labs', 'experience and background'],
  // one word
  ['internship', 'hiring availability'],
  ['resume', 'resume'],
  ['julia', 'programming background'],
  ['gpu', 'cuNumeric.jl'],
  ['email', 'email address'],
  ['linkedin', 'LinkedIn'],
  ['publications', 'publications and papers'],
  ['contact', 'contact'],
  ['skills', 'programming background'],
  ['availability', 'hiring availability'],
  ['pim', 'processing in memory'],
  ['legion', 'Legion runtime'],
  ['awards', 'awards'],
  ['location', 'contact'],
  // pasted recruiter messages
  [
    'hi david we are hiring a gpu performance engineer would you be interested',
    'hiring availability',
  ],
  [
    'hello i am a recruiter at a startup looking for distributed systems engineers',
    'hiring availability',
  ],
  ['we have an hpc internship for summer 2027 are you available', 'hiring availability'],
  ['i would like to schedule a quick call to discuss an opportunity', 'meeting'],
  // two intents
  ['what is his email and phone number', 'contact'],
  ['where is he located and is he willing to relocate', 'contact + logistics'],
  ['what does he research and where does he study', 'research interests + education'],
  // venues and fellowship
  ['what is ipdps', 'cuNumeric.jl'],
  ['what is asplos', 'PolymerPIM'],
  ['what is sc', 'presentations and talks'],
  ['what is the csgf', 'Department of Energy fellowship'],
  ['what is the doe csgf practicum', 'Department of Energy fellowship'],
  ['how long is his fellowship', 'Department of Energy fellowship'],
  ['what is juliacon', 'JuliaCon talk'],
  ['what is a technical report', 'publications and papers'],
  // working style and soft skills
  ['is he a good communicator', 'communication'],
  ['can he work independently', 'logistics'],
  ['has he written documentation', 'cuNumeric.jl'],
  ['does he work well with others', 'teamwork and leadership'],
  ['how does he approach problems', 'logistics'],
  ['has he managed a project', 'teamwork and leadership'],
  // evaluating strength
  ['what is his strongest project', 'projects'],
  ['what is his most impactful work', 'research interests'],
  ['how many citations does he have', 'Google Scholar'],
  ['has he won any competitions', 'awards'],
  ['is his work used by anyone', 'cuNumeric.jl'],
  ['does nvidia use his work', 'cuNumeric.jl'],
  // "and" questions that should split
  ['when does he graduate and is he looking for a job', 'graduation + hiring availability'],
  ['who is his advisor and what lab is he in', 'current role and university + Prescience Lab'],
  [
    'what are his skills and is he available for an internship',
    'programming background + hiring availability',
  ],
  ['tell me about cunumeric and polymerpim', 'cuNumeric.jl + PolymerPIM'],
  ['what awards has he won and what has he published', 'awards + publications and papers'],
  ['can i see his resume and schedule a meeting', 'resume + meeting'],
  [
    'where did he study and what did he do at los alamos',
    'education + Los Alamos National Laboratory',
  ],
  ['what is his linkedin & github', 'LinkedIn + GitHub profile'],
  // "and" questions that should stay one answer
  ['what is his email and phone number', 'contact'],
  ['what is his phone and email', 'contact'],
  ['does he know c++ and python', 'programming background'],
  ['what is vertical and horizontal fusion', 'PolymerPIM'],
  ['has he worked at both argonne and los alamos', 'experience and background'],
  ['does he know mpi and cuda', 'programming background'],
];

test('chat answers common questions with the right topic', async () => {
  const ask = loadChat();
  const wrong = [];
  for (const [question, topic] of EXPECTED) {
    const actual = await ask(question);
    if (actual !== topic) wrong.push(`"${question}": expected ${topic}, got ${actual}`);
  }
  assert.deepEqual(wrong, []);
});
