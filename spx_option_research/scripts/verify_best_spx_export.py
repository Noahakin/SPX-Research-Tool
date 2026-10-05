"""Verify folder inventory, PNG integrity, preservation hashes, and offline browser."""
from __future__ import annotations

import html
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from PIL import Image

from optimize_organized_spx_hedges import OUTPUT
from render_best_spx_hedges import protected_manifest
from render_organized_spx_charts import DEFAULT_OUTPUT, Library, CATEGORY_ORDER, TENORS, _browser_rows


ASSERTIONS = r"""<script>
(()=>{
const checks=[];
const check=(label,pass)=>checks.push({label,pass:Boolean(pass)});
const change=(id,value)=>{$(id).value=String(value);$(id).dispatchEvent(new Event('change',{bubbles:true}));};
const view=name=>document.querySelector('[data-view="'+name+'"]').click();
try {
check('complete library',data.length===8820&&byId.size===8820);
check('default is selected weekly 98/95 hedge',byId.get(state.selected).bestHedge&&byId.get(state.selected).primary===98&&byId.get(state.selected).width===3&&byId.get(state.selected).premium===.1&&byId.get(state.selected).tenor==='1 week');
change('category','Both');check('Both count',state.filtered.length===5880);
check('requested budgets',[...$('premium').options].map(o=>o.value).join(',')===',0.05,0.1,0.15,0.2');
change('premium',.1);check('budget count',state.filtered.length===1470);
change('primary',98);change('width',3);check('14 curves per short variation and budget',state.filtered.length===14);
change('hedgeType','Long put');check('seven expiries per hedge',state.filtered.length===7);
change('tenor','1 week');check('one selected winner',state.filtered.length===1);
const r=byId.get(state.selected);
view('base');check('short variation comparison',decodeURIComponent($('chart').getAttribute('src'))==='Both/10 percent premium/98-95/All strategies.webp');
view('hedge');check('budget comparison',decodeURIComponent($('chart').getAttribute('src'))==='Both/10 percent premium/All strategies.webp');
view('expiries');check('expiry comparison',decodeURIComponent($('chart').getAttribute('src'))==='Both/10 percent premium/98-95/Best long put/All expiries.webp');
view('single');check('individual selected hedge',decodeURIComponent($('chart').getAttribute('src'))===r.path);
check('retrospective selection disclosed',document.querySelector('.context').textContent.includes('retrospective winners'));
check('buffer protection rule disclosed',document.querySelector('.context').textContent.includes('Sharpe is reported but never used to select buffers'));
check('full buffer allocation disclosed',document.querySelector('.context').textContent.includes('spend the full allocation'));
check('budget tab label',document.querySelector('[data-view="hedge"]').textContent==='Premium budget');
change('category','Put selling');check('standalone category remains usable',state.filtered.length===1&&state.filtered[0].category==='Put selling');
$('reset').click();change('category','Put buying');check('put buying preserved',state.filtered.length===735);
$('reset').click();change('category','Both');change('premium',.1);change('primary',98);change('width',3);change('hedgeType','Long put');change('tenor','1 week');view('single');
change('premium',.2);change('hedgeType','Downside buffer');change('tenor','3 days');view('single');
check('corrected buffer image selected',decodeURIComponent($('chart').getAttribute('src'))==='Both/20 percent premium/98-95/Best put buffer/01 - 3 days.webp');
check('buffer title has exact budget',byId.get(state.selected).title.includes('(20% premium)')&&!byId.get(state.selected).title.includes('up to'));
} catch(error) { checks.push({label:String(error),pass:false}); }
setTimeout(()=>{check('selected image loads at full resolution',$('chart').complete&&$('chart').naturalWidth===1600&&$('chart').naturalHeight===1000);const output=document.createElement('pre');output.id='export-qa';output.textContent=JSON.stringify(checks);document.body.append(output);},1500);
})();</script>"""


def verify_browser(client):
    source = (client/"index.html").read_text(encoding="utf-8")
    instrumented = source.replace("<head>", '<head><base href="'+client.as_uri()+'/"><script>history.replaceState=function(){};</script>', 1)
    instrumented = instrumented.replace("</body>", ASSERTIONS+"</body>", 1)
    page = OUTPUT/"browser_qa.html"
    page.write_text(instrumented, encoding="utf-8")
    chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
    args = [str(chrome), "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        "--disable-background-networking", "--disable-component-update", "--disable-sync",
        "--disk-cache-size=1048576", "--media-cache-size=1048576",
        f"--user-data-dir={OUTPUT/'browser_qa_profile'}", "--window-size=1680,1400",
        "--virtual-time-budget=3000", "--dump-dom", page.as_uri()]
    run = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    if run.returncode:
        raise RuntimeError(run.stderr[-1000:])
    match = re.search(r'<pre id="export-qa">(.*?)</pre>', run.stdout, re.S)
    if not match:
        raise RuntimeError("Browser assertions did not finish")
    checks = json.loads(html.unescape(match.group(1)))
    assert all(row["pass"] for row in checks), checks
    return checks


def verify():
    # Inventory checks need metadata only. Numerical curves were independently
    # audited and their exact in-memory reconstruction was verified separately.
    config = json.loads((OUTPUT/"run.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(OUTPUT/"strategy_metrics.csv", dtype={"folder": str}, low_memory=False)
    metrics["_category_order"] = metrics.category.map({name: i for i, name in enumerate(CATEGORY_ORDER)})
    metrics["_tenor_index"] = metrics.tenor.map({name: i for i, name in enumerate(TENORS)})
    metrics["_hedge_notional_cap"] = config["hedge_notional_cap"]
    library = Library(config, metrics, np.array([]), np.empty((0, 0)), False)
    rows = _browser_rows(library)
    expected = {"All strategies.png"}
    for row in rows:
        for key in ("path", "expiryPath", "tenorPath", "categoryPath", "basePath", "hedgePath"):
            if row[key]:
                value = row[key]
                if row["category"] == "Both":
                    value = Path(value).with_suffix(".webp").as_posix()
                expected.add(value)
    actual = {p.relative_to(DEFAULT_OUTPUT).as_posix() for p in DEFAULT_OUTPUT.rglob("*") if p.suffix in (".png", ".webp")}
    assert expected == actual, {"missing": sorted(expected-actual)[:10], "extra": sorted(actual-expected)[:10]}
    assert len(actual) == 10537
    budgets = sorted(p.name for p in (DEFAULT_OUTPUT/"Both").iterdir() if p.is_dir())
    assert budgets == ["05 percent premium", "10 percent premium", "15 percent premium", "20 percent premium"]
    for budget in budgets:
        short_dirs = [p for p in (DEFAULT_OUTPUT/"Both"/budget).iterdir() if p.is_dir()]
        assert len(short_dirs) == 105
        for folder in short_dirs:
            assert sorted(p.name for p in folder.iterdir() if p.is_dir()) == ["Best long put", "Best put buffer"]
    for relative in sorted(actual):
        with Image.open(DEFAULT_OUTPUT/relative) as im:
            assert im.size == (1600, 1000), relative
            assert im.format == ("WEBP" if relative.endswith(".webp") else "PNG"), relative
            im.verify()
    before = json.loads((OUTPUT/"preserved_category_sha256.json").read_text(encoding="utf-8"))
    assert protected_manifest() == before, "Preserved category file hashes changed"
    checks = verify_browser(DEFAULT_OUTPUT)
    report = dict(status="passed", image_count=len(actual), image_dimensions=[1600, 1000],
        buffer_selection_rule_version="entry_affordability_v1",
        selected_hedges=5880, both_lossless_webp_count=sum(path.startswith("Both/") for path in actual),
        top_level_both_folders=budgets, short_variations_per_budget=105,
        preserved_files_sha256_verified=len(before), browser_checks=checks)
    (DEFAULT_OUTPUT/"Chart inventory validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTPUT/"export_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "browser_checks"}, indent=2), flush=True)


if __name__ == "__main__":
    verify()
