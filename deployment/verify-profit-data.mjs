// Reconstruct every exported variant in JavaScript against Python's daily NAV
// samples and independent full-period return statistics.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import '../spx_option_research/web/core.js';
import '../spx_option_research/web/profit-core.js';

const directory=resolve(process.argv[2]||'.publish/profit-taking-stage');
const extra=JSON.parse(readFileSync(join(directory,'metadata.json'),'utf8'));
const catalog=JSON.parse(readFileSync('SPX Research Interactive/catalog.js','utf8').replace(/^window\.SPX_CATALOG=/,'').replace(/;\s*$/,''));
catalog.dates=catalog.dates.slice(0,extra.protocol.daily_sessions);catalog.spx=catalog.spx.slice(0,extra.protocol.daily_sessions);
const P=globalThis.SPXProfit,M=globalThis.SPXMath;P.extendCatalog(catalog,extra);
const prices=new Map(),trades=new Map();
function read(key,meta){
  const script=readFileSync(join(directory,key+'.js'),'ascii');
  const match=/^window\.__SPX_CHUNK__\("([^"]+)","([^"]+)"\);\s*$/.exec(script);
  assert.equal(match?.[1],key);
  const packed=gunzipSync(Buffer.from(match[2],'base64'));
  assert.equal(packed.length,meta.bytes);
  assert.equal(createHash('sha256').update(packed).digest('hex'),meta.sha256);
  return P.unshuffle(packed.buffer.slice(packed.byteOffset,packed.byteOffset+packed.byteLength));
}
for(const [key,meta] of Object.entries(extra.priceChunks))prices.set(key,P.decodePrices(read(key,meta)));
const references=new Map(JSON.parse(gunzipSync(readFileSync(join(directory,'reference.json.gz')))).map(r=>[r.id,r]));
let count=0,maxDifference=0;
for(const r of catalog.strategies.filter(r=>r.profitTarget)){
  if(!trades.has(r.chunk))trades.set(r.chunk,P.decodeTrades(read(r.chunk,extra.chunks[r.chunk])));
  const nav=P.replay(r,trades.get(r.chunk),prices,catalog),reference=references.get(r.id);
  for(let mode=0;mode<2;mode++){
    for(let j=0;j<reference.indices.length;j++){
      const expected=reference.nav[mode][j],actual=nav[mode][reference.indices[j]],difference=Math.abs(actual-expected);
      maxDifference=Math.max(maxDifference,difference);
      assert(difference<=1e-8+Math.abs(expected)*1e-10,`${r.id}, mode ${mode}, session ${reference.indices[j]}: ${actual} vs ${expected}`);
    }
    if(reference.stats){
      const stats=M.statistics(nav[mode],0,catalog.dates.length-1,catalog.initial,catalog.dates);
      for(const [key,values] of Object.entries(reference.stats)){
        const expected=values[mode];
        if(expected===null)assert.equal(stats[key],null);
        else assert(Math.abs(stats[key]-expected)<=1e-8+Math.abs(expected)*1e-9,`${r.id} ${key}: ${stats[key]} vs ${expected}`);
      }
    }
  }
  if(++count%5000===0)console.log(`Verified ${count}/${references.size} profit-taking variants`);
}
assert.equal(count,references.size);
console.log(`PASS: ${count.toLocaleString()} variants, both exposures, all daily histories reconstructed; maximum sampled NAV difference $${maxDifference}.`);
