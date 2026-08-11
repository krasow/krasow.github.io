import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const terminalDirectory = path.dirname(fileURLToPath(import.meta.url));
const fileSystemDirectory = path.join(terminalDirectory, 'fs');
const manifestPath = path.join(fileSystemDirectory, 'manifest.json');
const textExtensions = new Set(['', '.css', '.html', '.js', '.json', '.md', '.pub', '.sh', '.txt']);

const walk = async (directory) => {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  const files = await Promise.all(entries
    .filter(({ name }) => name !== 'manifest.json')
    .map(async (entry) => {
      const absolutePath = path.join(directory, entry.name);
      return entry.isDirectory() ? walk(absolutePath) : [absolutePath];
    }));
  return files.flat();
};

const files = (await walk(fileSystemDirectory))
  .map((absolutePath) => {
    const relativePath = path.relative(fileSystemDirectory, absolutePath).split(path.sep).join('/');
    return {
      path: `/${relativePath}`,
      url: `/terminal/fs/${relativePath}`,
      text: textExtensions.has(path.extname(relativePath).toLowerCase()),
    };
  })
  .sort((left, right) => left.path.localeCompare(right.path));

await fs.writeFile(manifestPath, `${JSON.stringify({ files }, null, 2)}\n`);
console.log(`Included ${files.length} files in terminal/fs/manifest.json`);
