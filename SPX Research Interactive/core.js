(function(root){
  'use strict';
  const DAY=86400000,YEAR=365.2425;
  const time=s=>Date.parse(s+'T00:00:00Z');
  function lowerBound(a,v){let lo=0,hi=a.length;while(lo<hi){const m=(lo+hi)>>1;if(a[m]<v)lo=m+1;else hi=m;}return lo;}
  function windowIndices(dates,from,to){
    if(!from||!to||from>to)throw Error('Choose a start date on or before the end date.');
    const start=lowerBound(dates,from),end=lowerBound(dates,to+'\uffff')-1;
    if(start>end||start>=dates.length||end<0)throw Error('There are no research sessions in that date range.');
    return [start,Math.min(end,dates.length-1)];
  }
  function presetDates(dates,period){
    const end=dates[dates.length-1],d=new Date(time(end));
    if(period==='ALL')return [dates[0],end];
    if(period==='YTD')return [end.slice(0,4)+'-01-01',end];
    const n=Number(period.slice(0,-1)),unit=period.slice(-1),day=d.getUTCDate();
    d.setUTCDate(1);
    if(unit==='Y')d.setUTCFullYear(d.getUTCFullYear()-n);else d.setUTCMonth(d.getUTCMonth()-n);
    const last=new Date(Date.UTC(d.getUTCFullYear(),d.getUTCMonth()+1,0)).getUTCDate();
    d.setUTCDate(Math.min(day,last));
    return [d.toISOString().slice(0,10)<dates[0]?dates[0]:d.toISOString().slice(0,10),end];
  }
  function statistics(nav,start,end,initial,dates,includeDrawdown=false){
    const base=start?nav[start-1]:initial,n=end-start+1;
    let prev=base,peak=base,mean=0,m2=0,maxDD=0,min=Infinity,max=-Infinity;
    const dd=includeDrawdown?new Float32Array(n):null;
    for(let i=start;i<=end;i++){
      const value=nav[i],r=value/prev-1,count=i-start+1,delta=r-mean;
      mean+=delta/count;m2+=delta*(r-mean);peak=Math.max(peak,value);
      const draw=value/peak-1;maxDD=Math.min(maxDD,draw);if(dd)dd[i-start]=draw*100;
      min=Math.min(min,value/base*100);max=Math.max(max,value/base*100);prev=value;
    }
    const years=(time(dates[end])-time(dates[start]))/DAY/YEAR;
    const sd=n>1?Math.sqrt(Math.max(0,m2/(n-1))):NaN;
    return {base,total:nav[end]/base-1,cagr:years>0?Math.pow(nav[end]/base,1/years)-1:null,
      vol:Number.isFinite(sd)?sd*Math.sqrt(252):null,
      sharpe:Number.isFinite(sd)&&sd>1e-14?mean/sd*Math.sqrt(252):null,
      maxDD,min,max,drawdown:dd,observations:n};
  }
  function valueAt(curve,index,view){
    if(view==='drawdown')return curve.stats.drawdown[index-curve.start];
    const value=curve.nav[index]/curve.stats.base*100;
    return view==='return'?value-100:value;
  }
  function returnCorrelation(nav,benchmark,start,end,initial,benchmarkInitial=initial){
    const n=end-start+1;
    if(!nav||!benchmark||start<0||n<2||end>=nav.length||end>=benchmark.length)return null;
    let previous=start?nav[start-1]:initial,marketPrevious=start?benchmark[start-1]:benchmarkInitial;
    if(!Number.isFinite(previous)||previous<=0||!Number.isFinite(marketPrevious)||marketPrevious<=0)return null;
    let mean=0,marketMean=0,m2=0,marketM2=0,covariance=0;
    for(let i=start;i<=end;i++){
      const value=nav[i],market=benchmark[i];
      if(!Number.isFinite(value)||value<=0||!Number.isFinite(market)||market<=0)return null;
      const r=value/previous-1,s=market/marketPrevious-1,count=i-start+1;
      const delta=r-mean,marketDelta=s-marketMean;
      mean+=delta/count;marketMean+=marketDelta/count;
      m2+=delta*(r-mean);marketM2+=marketDelta*(s-marketMean);covariance+=delta*(s-marketMean);
      previous=value;marketPrevious=market;
    }
    // Constant daily returns have no defined correlation, including rounding noise.
    if(Math.sqrt(m2/(n-1))<=1e-14||Math.sqrt(marketM2/(n-1))<=1e-14)return null;
    const correlation=covariance/Math.sqrt(m2)/Math.sqrt(marketM2);
    return Number.isFinite(correlation)?Math.max(-1,Math.min(1,correlation)):null;
  }
  function niceTicks(lo,hi,count=5){
    const range=hi-lo||1,raw=range/count,power=10**Math.floor(Math.log10(raw)),fraction=raw/power;
    const step=(fraction<=1?1:fraction<=2?2:fraction<=2.5?2.5:fraction<=5?5:10)*power;
    const ticks=[];for(let v=Math.ceil(lo/step)*step;v<=hi+step*1e-6;v+=step)ticks.push(Math.abs(v)<step*1e-8?0:v);
    return {ticks,step};
  }
  function compareScores(a,b,descending=true){
    const validA=typeof a==='number'&&Number.isFinite(a),validB=typeof b==='number'&&Number.isFinite(b);
    if(!validA||!validB)return validA?-1:validB?1:0;
    return descending?b-a:a-b;
  }
  function rankingSource(catalog){
    return JSON.stringify([statistics.toString(),returnCorrelation.toString(),catalog.profitData?root.SPXProfit.signature:null,catalog.initial,catalog.dates,
      catalog.strategies.map(r=>[r.id,r.chunk,r.slot]),
      Object.entries(catalog.chunks).map(([key,c])=>[key,c.sha256,c.bytes])]);
  }
  const api={time,lowerBound,windowIndices,presetDates,statistics,returnCorrelation,valueAt,niceTicks,compareScores,rankingSource};
  root.SPXMath=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window==='undefined'?globalThis:window);
