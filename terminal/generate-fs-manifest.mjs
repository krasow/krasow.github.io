// Builds the terminal's filesystem manifest from the built site (_site/),
// so generated files (e.g. ~/publications) and page text match what's deployed.
// Runs after every Eleventy build; see eleventy.config.mjs.
import fs from 'node:fs/promises';
import path from 'node:path';
import { extractPageText } from './lib/page-text.mjs';

const pageTextUrl = (name) => `/assets/terminal/pages/${name}.txt`;

const textExtensions = new Set([
  '',
  '.css',
  '.html',
  '.js',
  '.json',
  '.md',
  '.pg',
  '.pub',
  '.sh',
  '.txt',
]);

const walk = async (directory) => {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const files = await Promise.all(
    entries
      .filter(({ name }) => name !== 'manifest.json')
      .map(async (entry) => {
        const absolutePath = path.join(directory, entry.name);
        return entry.isDirectory() ? walk(absolutePath) : [absolutePath];
      }),
  );
  return files.flat();
};

export const generateManifest = async (siteDirectory) => {
  const fileSystemDirectory = path.join(siteDirectory, 'terminal', 'fs');
  const manifestPath = path.join(fileSystemDirectory, 'manifest.json');
  // Pre-parsed page text, read by the browser terminal and the CLI so neither
  // parses HTML at runtime.
  const pageTextDirectory = path.join(siteDirectory, 'assets', 'terminal', 'pages');

  // Resolve a site-absolute URL (e.g. "/pages/about.html#experience") to a
  // built file, dropping any fragment.
  const sitePathToFile = (url) => path.join(siteDirectory, url.split('#')[0].replace(/^\//, ''));

  const writePageText = async (name, sourceUrl, selector) => {
    const html = await fs.readFile(sitePathToFile(sourceUrl), 'utf8');
    const text = extractPageText(html, selector);
    await fs.mkdir(pageTextDirectory, { recursive: true });
    await fs.writeFile(path.join(pageTextDirectory, `${name}.txt`), `${text}\n`);
  };

  const discovered = await walk(fileSystemDirectory);
  const directoryMetadata = discovered.filter(
    (absolutePath) => path.basename(absolutePath) === '.terminal.json',
  );
  const directories = directoryMetadata
    .map(
      (absolutePath) =>
        `/${path.relative(fileSystemDirectory, path.dirname(absolutePath)).split(path.sep).join('/')}`,
    )
    .sort((left, right) => left.localeCompare(right));

  const records = await Promise.all(
    discovered.map(async (absolutePath) => {
      const relativePath = path.relative(fileSystemDirectory, absolutePath).split(path.sep).join('/');
      if (path.basename(absolutePath) === '.terminal.json') {
        const metadata = JSON.parse(await fs.readFile(absolutePath, 'utf8'));
        const directory = path.posix.dirname(relativePath);
        return (metadata.entries ?? []).map(({ name, ...entry }) => ({
          path: `/${directory}/${name}`,
          text: false,
          ...entry,
        }));
      }
      if (relativePath.endsWith('.terminal')) {
        const [target, ...metadata] = (await fs.readFile(absolutePath, 'utf8'))
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean);
        const file = {
          path: `/${relativePath.slice(0, -'.terminal'.length)}`,
          target,
          text: false,
        };
        metadata.forEach((line) => {
          const separator = line.indexOf('=');
          if (separator > 0) file[line.slice(0, separator)] = line.slice(separator + 1);
        });
        return [file];
      }
      const file = {
        path: `/${relativePath}`,
        url: `/terminal/fs/${relativePath}`,
        text: textExtensions.has(path.extname(relativePath).toLowerCase()),
      };
      if (path.extname(relativePath) === '.pg') {
        const [target, ...metadata] = (await fs.readFile(absolutePath, 'utf8'))
          .split('\n')
          .map((line) => line.trim())
          .filter(Boolean);
        const options = {};
        metadata.forEach((line) => {
          const separator = line.indexOf('=');
          if (separator > 0) options[line.slice(0, separator)] = line.slice(separator + 1);
        });
        // Parse the page to text now; the manifest points `url` at the committed
        // result (read by `cat`) and keeps `target` for `open`/navigation.
        const name = path.basename(relativePath, '.pg');
        await writePageText(name, options.source ?? target, options.selector ?? '.holder');
        file.target = target;
        file.url = pageTextUrl(name);
      }
      return [file];
    }),
  );
  const files = records.flat();
  files.sort((left, right) => left.path.localeCompare(right.path));

  await fs.writeFile(manifestPath, `${JSON.stringify({ directories, files }, null, 2)}\n`);
  console.log(
    `Included ${directories.length} directories and ${files.length} files in terminal/fs/manifest.json`,
  );
};
