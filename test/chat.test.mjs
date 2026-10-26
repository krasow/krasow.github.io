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
  return async (question) => topics.get((await chat.ask(question)).answer) ?? FALLBACK;
};

// [question, topic of the answer it should get]
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
  ['does he know rust', FALLBACK],
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
  ['does he know openmp', FALLBACK],
  ['does he have experience with slurm', 'experience and background'],
  ['has he used supercomputers', 'hardware'],
  ['what hardware has he worked with', 'hardware'],
  ['does he do performance optimization', 'performance results'],
  ['has he done profiling or benchmarking', 'performance results'],
  ['does he know parallel programming', 'programming background'],
  ['does he have linux experience', 'experience and background'],
  ['does he know kubernetes', FALLBACK],
  ['does he have cloud experience', 'experience and background'],
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
  ['did he present at juliacon', 'presentations and talks'],
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
