// Builds the resume PDF from the site's content: renders
// content/documents/resume.11ty.js through Eleventy, compiles it with pdflatex,
// and copies the result to the PDF the site links to. Run with `npm run resume`.
import { execFileSync } from 'node:child_process';
import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { delimiter, join, resolve } from 'node:path';

import Eleventy from '@11ty/eleventy';

const root = resolve(import.meta.dirname, '..');
const build = join(root, 'resume/build');
const output = join(root, 'assets/documents/Krasowska_David_resume.pdf');

process.env.RESUME_TEX = '1';
const pages = await new Eleventy(join(root, 'content'), join(build, 'site'), {
  configPath: join(root, 'eleventy.config.mjs'),
  quietMode: true,
}).toJSON();
const page = pages.find((entry) => entry.url === '/resume/resume.tex');
if (!page) throw new Error('resume.11ty.js did not render');

mkdirSync(build, { recursive: true });
writeFileSync(join(build, 'resume.tex'), page.content);
try {
  execFileSync(
    'pdflatex',
    ['-interaction=nonstopmode', '-halt-on-error', '-output-directory', build, 'resume.tex'],
    {
      cwd: build,
      env: { ...process.env, TEXINPUTS: `${join(root, 'resume')}${delimiter}` },
      stdio: 'pipe',
    },
  );
} catch (error) {
  console.error(error.stdout?.toString().split('\n').slice(-30).join('\n'));
  throw new Error('pdflatex failed; full log in resume/build/resume.log');
}
copyFileSync(join(build, 'resume.pdf'), output);
console.log(`Wrote ${output.slice(root.length + 1)}`);
