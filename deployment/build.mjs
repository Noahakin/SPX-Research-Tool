import { copyFileSync, existsSync, linkSync, lstatSync, mkdirSync, readFileSync,
  readdirSync, realpathSync, rmSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import '../SPX Research Interactive/core.js';
import {readCatalog,readRankings} from './read-site-data.mjs';

const root = realpathSync(fileURLToPath(new URL('..', import.meta.url)));
const output = resolve(root, 'public');
const inputs = [
  'index.html',
  'README.md',
  'SPX Research Interactive',
  'SPX Research Organized Charts',
  'Profit Taking',
  'spx_option_research/README.md',
  'spx_option_research/results/dynamic_maturity_search',
  'spx_option_research/results/maturity_profit_grid',
];
const extensions = new Set(['html', 'css', 'js', 'json', 'md', 'csv', 'gz', 'png', 'webp', 'parquet']);

// Check the complete viewer dataset before replacing any previous build.
for (const input of inputs) {
  if (!existsSync(join(root, input))) throw new Error(`Missing deployment input: ${input}. Build from the repository root.`);
}
const viewer = join(root, 'SPX Research Interactive');
const catalog = readCatalog(viewer);
let rankings;
try {
  rankings = readRankings(join(viewer,'rankings.js'));
} catch {}
if (rankings?.version !== 3 || rankings.source !== globalThis.SPXMath.rankingSource(catalog)) {
  console.log('Refreshing rankings for the current chart data and statistics.');
  await import('./build-rankings.mjs');
}
for (const [key, chunk] of Object.entries(catalog.chunks)) {
  if (!/^(data\/c\d+|profit-data\/[pt]\d+)\.js$/.test(chunk.file)) throw new Error(`Unexpected data path for ${key}`);
  if (!existsSync(join(viewer, chunk.file)) || !statSync(join(viewer, chunk.file)).size) {
    throw new Error(`Missing chart data: ${chunk.file}. Deploy the complete GitHub repository.`);
  }
}

// Only this build's fixed output directory may be removed, never a linked folder.
if (dirname(output) !== root || output !== join(root, 'public')) throw new Error('Unsafe output directory');
if (existsSync(output)) {
  if (lstatSync(output).isSymbolicLink() || realpathSync(output) !== output) throw new Error('Refusing to replace a linked output directory');
  rmSync(output, { recursive: true });
}
mkdirSync(output, { recursive: true });

let files = 0, bytes = 0;
function publish(relative) {
  const source = join(root, relative), destination = join(output, relative);
  const info = lstatSync(source);
  if (info.isSymbolicLink()) throw new Error(`Unexpected linked deployment input: ${relative}`);
  if (info.isDirectory()) {
    for (const child of readdirSync(source).sort()) {
      if (child.startsWith('.') || child === '__pycache__' || /^(chrome|browser_qa)/i.test(child)) continue;
      publish(join(relative, child));
    }
    return;
  }
  if (!info.isFile() || !extensions.has(relative.split('.').pop().toLowerCase())) return;
  mkdirSync(dirname(destination), { recursive: true });
  // Share existing file storage where supported; Vercel receives regular files.
  try { linkSync(source, destination); }
  catch (error) {
    if (!['EXDEV', 'EPERM', 'EACCES', 'ENOTSUP', 'EMLINK'].includes(error.code)) throw error;
    copyFileSync(source, destination);
  }
  files++;
  bytes += info.size;
}
for (const input of inputs) publish(input);
console.log(`Static site built in public/: ${files} files, ${(bytes / 1024 ** 2).toFixed(1)} MiB.`);
console.log(`${catalog.strategies.length.toLocaleString('en-US')} strategies; ${Object.keys(catalog.chunks).length} data shards; both completed maturity studies.`);
