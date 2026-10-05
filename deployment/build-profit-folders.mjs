import {mkdirSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';

const root=join(process.cwd(),'Profit Taking');
const style='body{font:16px/1.65 system-ui,sans-serif;max-width:850px;margin:55px auto;padding:0 24px;color:#182b46;background:#f4f6fa}a{color:#3167d5}.folders{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:24px 0}.folders a{background:white;border:1px solid #dce4ef;border-radius:10px;padding:20px;font-weight:600}h1{line-height:1.2}p{color:#63758b}';
function page(title,body){return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+' · SPX Research</title><style>'+style+'</style><h1>'+title+'</h1>'+body+'</html>\n';}
mkdirSync(root,{recursive:true});
const targets=[25,50,75];
const intro='<p>Close all option legs when net profit reaches the target, then immediately enter a fresh trade. Put buying measures profit against the debit paid; Put selling and Both use net credit collected. Targets include entry and exit costs.</p>';
writeFileSync(join(root,'index.html'),page('Profit-taking folders',intro+'<div class="folders">'+targets.map(target=>'<a href="'+target+'%20percent/index.html">'+target+'% profit taking →</a>').join('')+'</div><p><a href="../SPX%20Research%20Interactive/index.html">Compare with hold to expiration</a> · <a href="../index.html">Research home</a></p>'));
for(const target of targets){
  const dir=join(root,target+' percent');mkdirSync(dir,{recursive:true});
  const viewer='../../SPX%20Research%20Interactive/index.html?profit='+(target/100);
  writeFileSync(join(dir,'index.html'),page(target+'% profit taking',intro+'<p><a href="'+viewer+'">Open all '+target+'% variants in the chart viewer →</a></p><div class="folders">'+['Put buying','Put selling','Both'].map(category=>'<a href="'+viewer+'&amp;category='+encodeURIComponent(category)+'">'+category+' →</a>').join('')+'</div><p>Every curve uses daily marks. With-SPX and options-only versions can be compared together over any date range. Each view starts with SPX alone.</p><p><a href="../index.html">All profit-taking folders</a> · <a href="../../index.html">Research home</a></p>'));
}
console.log('Created 25%, 50%, and 75% profit-taking folders.');
