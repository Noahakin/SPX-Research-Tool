// Precompute small ranking tables for the chart's standard date presets.
// Custom dates are calculated from daily NAV by the browser on demand.
import { readFileSync, writeFileSync, renameSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import '../SPX Research Interactive/core.js';

const M = globalThis.SPXMath;
const root = fileURLToPath(new URL('..', import.meta.url));
const site = join(root,'SPX Research Interactive');
const catalog = JSON.parse(readFileSync(join(site,'catalog.js'),'utf8').replace(/^window\.SPX_CATALOG=/,'').replace(/;\s*$/,''));
const ranges = new Map(['ALL','1M','3M','6M','YTD','1Y','3Y','5Y'].map(period => {
  const range = M.windowIndices(catalog.dates,...M.presetDates(catalog.dates,period));
  return [range.join(':'),range];
}));
const windows = Object.fromEntries([...ranges.keys()].map(key => [key,new Array(catalog.strategies.length*4).fill(null)]));
const groups = new Map();
catalog.strategies.forEach((r,i) => {
  if (!groups.has(r.chunk)) groups.set(r.chunk,[]);
  groups.get(r.chunk).push({record:r,index:i});
});
let completed = 0;
for (const [key,chunk] of Object.entries(catalog.chunks)) {
  const script = readFileSync(join(site,chunk.file),'ascii');
  const match = /^window\.__SPX_CHUNK__\("([^"]+)","([^"]+)"\);\s*$/.exec(script);
  if (!match || match[1] !== key) throw new Error(`Invalid data shard: ${key}`);
  const raw = gunzipSync(Buffer.from(match[2],'base64'));
  if (raw.length !== chunk.bytes || createHash('sha256').update(raw).digest('hex') !== chunk.sha256) throw new Error(`Data integrity check failed: ${key}`);
  const data = raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength), n = catalog.dates.length;
  for (const {record:r,index:i} of groups.get(key) || []) {
    for (const mode of [0,1]) {
      const nav = new Float64Array(data,(r.slot*2+mode)*n*8,n);
      for (const [range,[start,end]] of ranges) {
        const stats = M.statistics(nav,start,end,catalog.initial,catalog.dates);
        windows[range][i*4+mode*2] = Number.isFinite(stats.cagr) ? stats.cagr : null;
        windows[range][i*4+mode*2+1] = Number.isFinite(stats.sharpe) ? stats.sharpe : null;
      }
    }
  }
  if (++completed % 32 === 0) console.log(`Ranking tables: ${completed}/${Object.keys(catalog.chunks).length} verified data shards`);
}
const artifact = 'window.SPX_RANKINGS='+JSON.stringify({version:1,source:M.rankingSource(catalog),windows})+';\n';
const pending = join(site,'rankings.js.tmp');
writeFileSync(pending,artifact); renameSync(pending,join(site,'rankings.js'));
console.log(`Rankings ready: ${catalog.strategies.length} strategies, both exposures, ${ranges.size} periods, ${(Buffer.byteLength(artifact)/1024**2).toFixed(1)} MiB.`);
