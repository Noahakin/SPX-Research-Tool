// Precompute small ranking tables for the chart's standard date presets.
// Custom dates are calculated from daily NAV by the browser on demand.
import { readFileSync, writeFileSync, renameSync } from 'node:fs';
import { gzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import '../SPX Research Interactive/core.js';
import {readCatalog,readChunk} from './read-site-data.mjs';

const M = globalThis.SPXMath;
const root = fileURLToPath(new URL('..', import.meta.url));
const site = join(root,'SPX Research Interactive');
const catalog = readCatalog(site), P=globalThis.SPXProfit, prices=new Map();
const ranges = new Map(['ALL','1M','3M','6M','YTD','1Y','3Y','5Y'].map(period => {
  const range = M.windowIndices(catalog.dates,...M.presetDates(catalog.dates,period));
  return [range.join(':'),range];
}));
const windows = Object.fromEntries([...ranges.keys()].map(key => [key,new Array(catalog.strategies.length*6).fill(null)]));
const groups = new Map();
catalog.strategies.forEach((r,i) => {
  if (!groups.has(r.chunk)) groups.set(r.chunk,[]);
  groups.get(r.chunk).push({record:r,index:i});
});
let completed = 0;
for (const [key,chunk] of Object.entries(catalog.chunks)) {
  if(chunk.kind==='prices')continue;
  const data = readChunk(site,key,chunk), n = catalog.dates.length;
  let trades;
  if(chunk.kind==='trades'){
    trades=P.decodeTrades(data);
    for(const price of chunk.prices)if(!prices.has(price))prices.set(price,P.decodePrices(readChunk(site,price,catalog.chunks[price])));
  }
  for (const {record:r,index:i} of groups.get(key) || []) {
    const replay=r.profitTarget?P.replay(r,trades,prices,catalog):null;
    for (const mode of [0,1]) {
      const nav = replay?replay[mode]:new Float64Array(data,(r.slot*2+mode)*n*8,n);
      for (const [range,[start,end]] of ranges) {
        const stats = M.statistics(nav,start,end,catalog.initial,catalog.dates);
        windows[range][i*6+mode*3] = Number.isFinite(stats.cagr) ? stats.cagr : null;
        windows[range][i*6+mode*3+1] = Number.isFinite(stats.sharpe) ? stats.sharpe : null;
        windows[range][i*6+mode*3+2] = Number.isFinite(stats.maxDD) ? stats.maxDD : null;
      }
    }
  }
  if (++completed % 32 === 0) console.log(`Ranking tables: ${completed}/${Object.keys(catalog.chunks).length} verified data shards`);
}
const packed=gzipSync(Buffer.from(JSON.stringify({version:2,source:M.rankingSource(catalog),windows})));
const artifact = 'window.SPX_RANKINGS_COMPRESSED='+JSON.stringify(packed.toString('base64'))+';\n';
const pending = join(site,'rankings.js.tmp');
writeFileSync(pending,artifact); renameSync(pending,join(site,'rankings.js'));
console.log(`Rankings ready: ${catalog.strategies.length} strategies, both exposures, ${ranges.size} periods, ${(Buffer.byteLength(artifact)/1024**2).toFixed(1)} MiB.`);
