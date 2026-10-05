import {readFileSync,writeFileSync,existsSync,renameSync,realpathSync} from 'node:fs';
import {join,resolve,dirname} from 'node:path';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';

const root=realpathSync(process.cwd());
const source=resolve(root,'.publish/profit-taking-stage');
const destination=resolve(root,'SPX Research Interactive/profit-data');
assert.equal(dirname(source),join(root,'.publish'));
assert.equal(dirname(destination),join(root,'SPX Research Interactive'));
assert.equal(realpathSync(source),source,'Refuse to move a linked source directory');
assert(!existsSync(destination),'Profit-taking data already installed; review before replacing it');
const metadata=JSON.parse(readFileSync(join(source,'metadata.json'),'utf8'));
assert.equal(metadata.records.length,22050);
assert.equal(metadata.protocol.daily_sessions,2510);
assert.deepEqual([...new Set(metadata.records.map(r=>r.profitTarget))].sort(),[.25,.5,.75]);
const protocol={...metadata.protocol,engine_sha256:createHash('sha256').update(readFileSync('spx_option_research/scripts/profit_taking_research.py')).digest('hex')};
protocol.entry_eligibility='Nearest listed strikes are selected before screening, with maximum deviation 0.5 percentage point from each target. Entries require valid observed quotes, positive net credit for selling/Both or positive debit for buying, distinct spread strikes, and midpoints within vertical payoff bounds.';
protocol.valuation_calibration='Isolated bad held quotes use same-day adjacent puts or calibrated put-call parity. If an extremely wide donor spoils the parity fit, broad donor markets are excluded and all remaining calibration safeguards are retained. When an ask is missing but its bid is valid, the bid is retained and the spread is estimated from valid bracketing puts in the same snapshot and expiry, subject to donor-distance, monotonicity, and payoff checks. Estimates never authorize execution.';
protocol.quote_estimate_scope='The estimate count covers distinct contract/date marks queried for held positions or entry screening. Each variant also records how many held days used an estimate.';
writeFileSync(join(source,'Protocol.json'),JSON.stringify(protocol,null,2)+'\n');
renameSync(source,destination);
writeFileSync(join(root,'SPX Research Interactive/profit-catalog.js'),'window.SPX_PROFIT_DATA='+JSON.stringify(metadata)+';\n');
const validationPath=join(root,'SPX Research Interactive/Data validation.json');
const validation=JSON.parse(readFileSync(validationPath,'utf8'));
validation.profit_taking={status:'passed',variants:metadata.records.length,exposure_lines:metadata.records.length*2,
  daily_sessions:metadata.protocol.daily_sessions,trades:metadata.protocol.trades,
  shared_contracts:metadata.protocol.contracts,targets:metadata.protocol.targets,
  validation:'Accounting tests and complete JavaScript replay versus Python daily NAV samples and full-period statistics, both exposures.',
  protocol:'profit-data/Protocol.json'};
validation.total_option_strategies=29400;validation.total_exposure_lines=58800;
validation.total_daily_values=58800*metadata.protocol.daily_sessions;
writeFileSync(validationPath,JSON.stringify(validation,null,2)+'\n');
const readme=join(root,'SPX Research Interactive/README.md');
writeFileSync(readme,readFileSync(readme,'utf8').replace('The site includes 7,350 distinct strategies and 14,700 options-only/SPX combinations, plus SPX as a benchmark.',
  'The site includes 7,350 original strategies plus 22,050 profit-taking variants: 29,400 strategies and 58,800 options-only/SPX combinations, plus SPX as a benchmark.')+
  '\nSelect a **Strategy folder** for hold to expiration or profit taking at **25%, 50%, or 75%**. All option legs close together at the net-profit target, followed by a fresh entry at the same daily snapshot. Put buying measures profit against the initial debit; Put selling and Both use retained net credit. SPX remains invested. Entry and early exit both include costs. See [Protocol.json](profit-data/Protocol.json) for the full rules.\n');
console.log('Installed 22,050 profit-taking variants with shared daily prices and trade records.');
