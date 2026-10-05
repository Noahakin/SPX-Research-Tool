(function(root){
  'use strict';
  function extendCatalog(catalog,extra){
    if(!extra)return catalog;
    if(extra.version!==1)throw Error('Unsupported profit-taking catalog');
    if(catalog.profitData)return catalog;
    const originals=new Map(catalog.strategies.map(r=>[r.id,r]));
    for(const item of extra.records){
      const base=originals.get(item.base);if(!base)throw Error('Unknown profit-taking base strategy');
      catalog.strategies.push({...base,...item,baseId:item.base,
        subtitle:base.subtitle+' · Take '+Math.round(item.profitTarget*100)+'%',
        cycles:item.trades,cashCycles:0,metrics:null});
    }
    Object.assign(catalog.chunks,extra.chunks,extra.priceChunks);
    catalog.profitData=extra;return catalog;
  }
  function unshuffle(buffer){
    const bytes=new Uint8Array(buffer),length=bytes.length;
    if(length%8)throw Error('Invalid shuffled data length');
    const result=new Uint8Array(length),rows=length/8;
    for(let byte=0;byte<8;byte++)for(let row=0;row<rows;row++)result[row*8+byte]=bytes[byte*rows+row];
    return result.buffer;
  }
  function decodeTrades(buffer){
    const values=new Float64Array(buffer),[version,count,trades]=values;
    if(version!==1||values.length!==4+count+trades*7)throw Error('Invalid trade data');
    return {values,count,trades,offsets:values.subarray(3,4+count),start:4+count};
  }
  function decodePrices(buffer){
    const values=new Float64Array(buffer),[version,first,count,length]=values;
    if(version!==1||values.length!==4+count*5+length)throw Error('Invalid daily price data');
    return {values,first,count,start:4+count*5};
  }
  function replay(record,trades,prices,catalog){
    const n=catalog.dates.length,spx=catalog.spx,initial=catalog.initial;
    const result=[new Float64Array(n),new Float64Array(n)];
    const epoch=catalog.dates.map(d=>Date.parse(d+'T00:00:00Z')/86400000);
    const values=trades.values,begin=trades.offsets[record.slot],end=trades.offsets[record.slot+1];
    let equity0=initial,equity1=initial,anchor=0,last=-1;
    const pathFor=id=>{
      if(id<0)return null;
      const key='p'+Math.floor(id/catalog.profitData.priceGroup).toString().padStart(3,'0');
      const path=prices.get(key);if(!path)throw Error('Missing profit-taking daily prices: '+key);
      const at=4+(id-path.first)*5;
      return {values:path.values,first:path.values[at],length:path.values[at+1],start:path.start+path.values[at+2],strike:path.values[at+3],expiry:path.values[at+4]};
    };
    const quote=(path,day)=>{
      if(!path)return [0,0,1];
      if(path.expiry===epoch[day])return [Math.max(path.strike-spx[day],0),0,1];
      const offset=day-path.first,at=path.start+offset*3;
      if(offset<0||offset>=path.length||!Number.isFinite(path.values[at])||!Number.isFinite(path.values[at+1])||!path.values[at+2])throw Error('Missing held daily mark');
      return [path.values[at],path.values[at+1],path.values[at+2]];
    };
    for(let trade=begin;trade<end;trade++){
      const at=trades.start+trade*7,entry=values[at],exit=values[at+1],reason=values[at+6];
      if(entry<last||exit<entry||exit>=n)throw Error('Invalid profit-taking trade dates');
      for(let day=last+1;day<entry;day++){
        result[0][day]=equity0;result[1][day]=equity1*spx[day]/spx[anchor];
      }
      equity1*=spx[entry]/spx[anchor];
      const paths=[0,1,2,3].map(leg=>pathFor(values[at+2+leg]-1));
      const opening=paths.map(path=>quote(path,entry));
      const side=record.category==='Put buying'?1:-1;
      const qty=[side,paths[1]?-side:0,0,0];
      if(record.category==='Both'){
        const shortCredit=opening[0][0]-opening[0][1]-(paths[1]?opening[1][0]+opening[1][1]:0);
        const hedgeDebit=opening[2][0]+opening[2][1]-(paths[3]?opening[3][0]-opening[3][1]:0);
        const ratio=record.premium*shortCredit/hedgeDebit;
        qty[2]=record.dynamic?ratio:Math.min(1,ratio);qty[3]=paths[3]?-qty[2]:0;
      }
      let credit=0;
      for(let leg=0;leg<4;leg++)credit-=qty[leg]*opening[leg][0]+Math.abs(qty[leg])*opening[leg][1];
      const before0=equity0,before1=equity1,spot=spx[entry];
      for(let day=entry;day<=exit;day++){
        let pnl=credit;
        for(let leg=0;leg<4;leg++)if(qty[leg]){
          const value=quote(paths[leg],day);pnl+=qty[leg]*value[0];
          if(day===exit&&reason===1)pnl-=Math.abs(qty[leg])*value[1];
        }
        result[0][day]=before0*(1+pnl/spot);
        result[1][day]=before1*(1+pnl/spot)+before1*(spx[day]/spot-1);
      }
      equity0=result[0][exit];equity1=result[1][exit];anchor=exit;last=exit;
    }
    for(let day=last+1;day<n;day++){
      result[0][day]=equity0;result[1][day]=equity1*spx[day]/spx[anchor];
    }
    return result;
  }
  const api={extendCatalog,unshuffle,decodeTrades,decodePrices,replay,
    signature:JSON.stringify([replay.toString(),decodePrices.toString(),decodeTrades.toString()])};
  root.SPXProfit=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window==='undefined'?globalThis:window);
