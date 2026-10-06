const test = require('node:test');
const assert = require('node:assert/strict');
const {compareScores,statistics} = require('../web/core.js');

test('unavailable scores stay last in either direction, including zero-volatility Sharpe', () => {
  const flat = statistics([100,100,100],0,2,100,['2020-01-01','2020-01-02','2020-01-03']);
  assert.equal(flat.sharpe,null);
  const rows=[{id:'flat',v:flat.sharpe},{id:'loss',v:-1},{id:'gain',v:2},{id:'zero',v:0},{id:'invalid',v:NaN}];
  assert.deepEqual(rows.slice().sort((a,b)=>compareScores(a.v,b.v)).map(r=>r.id),['gain','zero','loss','flat','invalid']);
  assert.deepEqual(rows.slice().sort((a,b)=>compareScores(a.v,b.v,false)).map(r=>r.id),['loss','zero','gain','flat','invalid']);
});

test('rankings change with the selected dates and preserve genuine ties', () => {
  const dates=['2020-01-01','2020-01-02','2020-01-03','2020-01-06'];
  const early=[105,110,108,106],late=[99,98,101,105];
  const first=[early,late].map(nav=>statistics(nav,0,1,100,dates));
  const last=[early,late].map(nav=>statistics(nav,2,3,100,dates));
  assert(compareScores(first[0].cagr,first[1].cagr)<0);
  assert(compareScores(last[0].cagr,last[1].cagr)>0);
  assert(compareScores(first[0].sharpe,first[1].sharpe)<0);
  assert(compareScores(last[0].sharpe,last[1].sharpe)>0);
  assert.equal(compareScores(1.25,1.25),0);
});

test('a single-session window has no annualized CAGR or Sharpe to rank', () => {
  const stats=statistics([100,105],1,1,100,['2020-01-01','2020-01-02']);
  assert.equal(stats.cagr,null);assert.equal(stats.sharpe,null);
  assert.equal(compareScores(stats.cagr,0),1);
});

test('smallest drawdown ranks zero and shallow losses before deep losses', () => {
  const dates=['2020-01-01','2020-01-02','2020-01-03'];
  const rows=[
    {id:'deep',nav:[100,80,100]},
    {id:'shallow',nav:[100,95,100]},
    {id:'rising',nav:[100,105,110]},
  ].map(r=>({id:r.id,v:statistics(r.nav,0,2,100,dates).maxDD}));
  rows.push({id:'missing',v:null});
  assert.deepEqual(rows.slice().sort((a,b)=>compareScores(a.v,b.v)).map(r=>r.id),['rising','shallow','deep','missing']);
  assert.deepEqual(rows.slice().sort((a,b)=>compareScores(a.v,b.v,false)).map(r=>r.id),['deep','shallow','rising','missing']);
});

test('drawdown rankings use the selected window including its first-session loss', () => {
  const dates=['2020-01-01','2020-01-02','2020-01-03','2020-01-06'];
  const recovered=[100,80,100,110],lateLoss=[100,100,95,100];
  const full=[recovered,lateLoss].map(nav=>statistics(nav,0,3,100,dates).maxDD);
  const recent=[recovered,lateLoss].map(nav=>statistics(nav,2,3,100,dates).maxDD);
  assert(compareScores(full[0],full[1])>0);
  assert(compareScores(recent[0],recent[1])<0);
  assert(Math.abs(recent[1]+0.05)<1e-12);
  const oneDay=statistics(lateLoss,2,2,100,dates);
  assert(Math.abs(oneDay.maxDD+0.05)<1e-12);
  assert.equal(oneDay.cagr,null);
});
