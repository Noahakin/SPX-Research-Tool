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
  function niceTicks(lo,hi,count=5){
    const range=hi-lo||1,raw=range/count,power=10**Math.floor(Math.log10(raw)),fraction=raw/power;
    const step=(fraction<=1?1:fraction<=2?2:fraction<=2.5?2.5:fraction<=5?5:10)*power;
    const ticks=[];for(let v=Math.ceil(lo/step)*step;v<=hi+step*1e-6;v+=step)ticks.push(Math.abs(v)<step*1e-8?0:v);
    return {ticks,step};
  }
  const api={time,lowerBound,windowIndices,presetDates,statistics,valueAt,niceTicks};
  root.SPXMath=api;if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window==='undefined'?globalThis:window);
