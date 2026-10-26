// site.personalPhone is for the resume PDF only (resume/build.mjs); nothing the
// website builds may contain it. Runs against _site, so run `npm test`.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import site from '../content/_data/site.json' with { type: 'json' };

const siteDirectory = new URL('../_site/', import.meta.url).pathname;
const TEXT = /\.(html|json|txt|md|xml|vcf|js|css|pg|tex)$/;

const files = (directory) =>
  fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(directory, entry.name);
    return entry.isDirectory() ? files(full) : [full];
  });

test('the personal phone number never reaches the website', () => {
  // The last ten digits, in any format: 8432837758, (843) 283-7758, 843.283.7758…
  const digits = site.personalPhone.replace(/\D/g, '').slice(-10);
  const number = new RegExp(digits.split('').join('\\D{0,3}'));
  const leaks = files(siteDirectory)
    .filter((file) => TEXT.test(file))
    .filter((file) => number.test(fs.readFileSync(file, 'utf8')))
    .map((file) => path.relative(siteDirectory, file));
  assert.deepEqual(leaks, []);
});
