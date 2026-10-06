const test = require('node:test');
const assert = require('node:assert/strict');
const {returnCorrelation} = require('../web/core.js');

function nav(returns, initial=100) {
  let value=initial;
  return returns.map(r=>value*=1+r);
}
function close(actual, expected) {
  assert.notEqual(actual,null);
  assert(Math.abs(actual-expected)<1e-12,`${actual} should equal ${expected}`);
}

test('correlation uses daily returns, with known positive, negative, and zero relationships',()=>{
  const market=[.01,.02,.03,.04], path=nav(market);
  close(returnCorrelation(path,path,0,3,100),1);
  close(returnCorrelation(nav(market.map(r=>-2*r)),path,0,3,100),-1);
  close(returnCorrelation(nav([.01,.03,.02,.04]),path,0,3,100),.8);
  close(returnCorrelation(nav([.01,-.01,.01,-.01]),nav([.01,.01,-.01,-.01]),0,3,100),0);
});

test('correlation uses the prior close for a selected window and includes its first return',()=>{
  const strategy=nav([.4,-.3,.01,.03,.02,.04,.6]);
  const market=nav([-.2,.3,.01,.02,.03,.04,-.5]);
  close(returnCorrelation(strategy,market,2,5,100),.8);
  close(returnCorrelation(strategy,market,4,5,100),1);
  assert(Math.abs(returnCorrelation(strategy,market,0,6,100)-.8)>.1);
});

test('correlation is independent of portfolio scale and accepts a separate benchmark initial value',()=>{
  const a=[.01,.03,.02,.04],b=[.01,.02,.03,.04];
  close(returnCorrelation(nav(a,1000000),nav(b,4000),0,3,1000000,4000),.8);
  close(returnCorrelation(nav(b,4000),nav(a,1000000),0,3,4000,1000000),.8);
});

test('one observation, flat or constant daily returns, and invalid data are unavailable',()=>{
  const moving=nav([.01,-.02,.03,.02]);
  assert.equal(returnCorrelation(moving,moving,2,2,100),null);
  assert.equal(returnCorrelation([100,100,100,100],moving,0,3,100),null);
  assert.equal(returnCorrelation(moving,[100,100,100,100],0,3,100),null);
  assert.equal(returnCorrelation(nav([.01,.01,.01,.01]),moving,0,3,100),null);
  assert.equal(returnCorrelation([100,NaN,101,102],moving,0,3,100),null);
  assert.equal(returnCorrelation(moving,[100,101],0,3,100),null);
  assert.equal(returnCorrelation([0,101,102,103],moving,1,3,100),null);
});
