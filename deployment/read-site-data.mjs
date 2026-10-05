import {readFileSync,existsSync} from 'node:fs';
import {join} from 'node:path';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import '../SPX Research Interactive/profit-core.js';

export function readCatalog(site){
  const catalog=JSON.parse(readFileSync(join(site,'catalog.js'),'utf8').replace(/^window\.SPX_CATALOG=/,'').replace(/;\s*$/,''));
  if(existsSync(join(site,'profit-catalog.js'))){
    const extra=JSON.parse(readFileSync(join(site,'profit-catalog.js'),'utf8').replace(/^window\.SPX_PROFIT_DATA=/,'').replace(/;\s*$/,''));
    globalThis.SPXProfit.extendCatalog(catalog,extra);
  }
  return catalog;
}
export function readRankings(path){
  const text=readFileSync(path,'utf8');
  if(text.startsWith('window.SPX_RANKINGS_COMPRESSED=')){
    const encoded=JSON.parse(text.replace(/^window\.SPX_RANKINGS_COMPRESSED=/,'').replace(/;\s*$/,''));
    return JSON.parse(gunzipSync(Buffer.from(encoded,'base64')));
  }
  return JSON.parse(text.replace(/^window\.SPX_RANKINGS=/,'').replace(/;\s*$/,''));
}
export function readChunk(site,key,chunk){
  const script=readFileSync(join(site,chunk.file),'ascii');
  const match=/^window\.__SPX_CHUNK__\("([^"]+)","([^"]+)"\);\s*$/.exec(script);
  if(!match||match[1]!==key)throw Error('Invalid data shard: '+key);
  const raw=gunzipSync(Buffer.from(match[2],'base64'));
  if(raw.length!==chunk.bytes||createHash('sha256').update(raw).digest('hex')!==chunk.sha256)throw Error('Data integrity check failed: '+key);
  const buffer=raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength);
  return chunk.encoding==='shuffle8'?globalThis.SPXProfit.unshuffle(buffer):buffer;
}
