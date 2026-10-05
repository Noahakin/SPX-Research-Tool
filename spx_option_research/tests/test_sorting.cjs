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
