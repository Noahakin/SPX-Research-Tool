(function () {
  'use strict';
  const C = window.SPX_CATALOG, M = window.SPXMath, $ = id => document.getElementById(id);
  if (!C || !M || !window.DecompressionStream) {
    $('status').textContent = 'Open the complete website folder in current Chrome or Edge.';
    $('chartProgress').hidden = false;
    $('chartProgress').textContent = 'The data files are missing or this browser needs an update.';
    return;
  }
  const dates = C.dates, N = dates.length, byId = new Map(C.strategies.map(r => [r.id, r]));
  const modes = ['options', 'spx', 'both'], filterIds = ['tenor', 'primary', 'width', 'premium', 'hedge'];
  const palette = ['#3269cf', '#129487', '#cc8850', '#965db6', '#d3657c', '#439ab7', '#6c8750', '#ae683f'];
  const state = {selected: new Map(), filtered: [], category: '', start: 0, end: N - 1,
    view: 'growth', benchmark: true, busy: false, ready: false, curves: [], error: null};
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const count = n => n.toLocaleString('en-US');
  const fmt = (v, digits = 1) => v == null || !Number.isFinite(v) ? '—' :
    Math.abs(v) >= 1e7 ? v.toExponential(2) : v.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});
  const pct = v => v == null ? '—' : fmt(v * 100) + '%';
  const shortNumber = v => new Intl.NumberFormat('en-US', {notation: 'compact', maximumFractionDigits: 1}).format(v);
  const friendlyDate = s => new Date(M.time(s)).toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric', timeZone:'UTC'});
  const frame = () => new Promise(resolve => requestAnimationFrame(resolve));
  const lineModes = mode => mode === 'both' ? ['options', 'spx'] : [mode];
  const modeName = mode => mode === 'spx' ? 'With SPX' : 'Options only';
  let colorIndex = 0, revision = 0, refreshTimer, toastTimer, plot = null, drag = null, hoverIndex = null;
  let selectedIds = [], metrics = new Map(), metricWindow = '', hoverRow = null;
  const catalogOrder = new Map(C.strategies.map((r,i) => [r.id,i]));
  let sortRevision = 0, sortBusy = false, sortError = null, sortWindow = '', sortRange = '';
  let sortScores = new Map(), rankingIndexPromise;
  let chartMetricRows = [], chartMetricRowHeight = 48;
  const chartMetricKeys = ['cagr','vol','maxDD','sharpe'];
  const spxNav = Float64Array.from(C.spx, v => C.initial * v / C.spx[0]);
  const searchText = new Map(C.strategies.map(r => [r.id,
    [r.name, r.subtitle, r.category, r.hedge, r.days + ' days', r.hedgeHigh && 'long ' + r.hedgeHigh].join(' ').toLowerCase()]));
  function detail(r) {
    return r.subtitle + (r.hedge === 'Long put' ? ' · Buy ' + r.hedgeHigh + '% put' : r.dynamic ? ' · 3–5pt buffer' : '');
  }
  function description(r) {
    return r.name + ' · ' + detail(r) + (r.dynamic ? ' · Closest affordable target, full credit allocation' : '') +
      ' · ' + count(r.cashCycles) + ' cash cycles of ' + count(r.cycles);
  }
  function toast(message) {
    $('toast').textContent = message; $('toast').hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 5500);
  }
  function color() {
    const i = colorIndex++; if (i < palette.length) return palette[i];
    const h = (i * 137.508) % 360, s = .54, l = .45;
    const a = s * Math.min(l, 1-l), f = n => { const k = (n + h/30) % 12; return l - a * Math.max(-1, Math.min(k-3, 9-k, 1)); };
    return '#' + [f(0), f(8), f(4)].map(v => Math.round(v*255).toString(16).padStart(2,'0')).join('');
  }
  // Script shards load over file://. Decompressed arrays retain all daily data.
  const cache = new Map(), loading = new Map(), jobs = new Map(), queue = [];
  let activeLoads = 0;
  function loadChunk(key) {
    if (cache.has(key)) return Promise.resolve(cache.get(key));
    if (loading.has(key)) return loading.get(key);
    const promise = new Promise((resolve, reject) => queue.push({key, resolve, reject}));
    loading.set(key, promise); pump(); return promise;
  }
  function pump() {
    while (activeLoads < 3 && queue.length) {
      const job = queue.shift(); activeLoads++; jobs.set(job.key, job);
      const script = document.createElement('script'); job.script = script; script.src = C.chunks[job.key].file;
      script.onerror = () => finish(job, Error('Could not load ' + script.getAttribute('src') + '. Keep the data folder with index.html.'));
      job.timer = setTimeout(() => finish(job, Error('Data load timed out. Reopen the page and try again.')), 90000);
      document.head.append(script);
    }
  }
  function finish(job, error, buffer) {
    if (!jobs.has(job.key)) return;
    clearTimeout(job.timer); job.script.remove(); jobs.delete(job.key); activeLoads--;
    if (error) { loading.delete(job.key); job.reject(error); }
    else { cache.set(job.key, buffer); job.resolve(buffer); }
    pump();
  }
  window.__SPX_CHUNK__ = async function (key, encoded) {
    const job = jobs.get(key); if (!job) return;
    try {
      const binary = atob(encoded), bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      const buffer = await new Response(stream).arrayBuffer();
      if (buffer.byteLength !== C.chunks[key].bytes) throw Error('Incorrect data length in ' + key);
      if (crypto.subtle) {
        const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buffer)), v => v.toString(16).padStart(2,'0')).join('');
        if (hash !== C.chunks[key].sha256) throw Error('Data integrity check failed for ' + key);
      }
      finish(job, null, buffer);
    } catch (error) { finish(job, error); }
  };
  function navFor(id, mode) {
    if (id === 'benchmark') return spxNav;
    const r = byId.get(id), buffer = cache.get(r.chunk);
    return buffer ? new Float64Array(buffer, (r.slot * 2 + (mode === 'spx' ? 1 : 0)) * N * 8, N) : null;
  }
  function filter() {
    const terms = $('search').value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    state.filtered = C.strategies.filter(r => (!state.category || r.category === state.category) &&
      filterIds.every(id => !$(id).value || String(r[id]) === $(id).value) && terms.every(t => searchText.get(r.id).includes(t)));
    $('resultCount').textContent = count(state.filtered.length) + ' strategies';
    $('strategyList').scrollTop = 0; sortLibrary();
  }
  function loadRankingIndex() {
    if (rankingIndexPromise) return rankingIndexPromise;
    rankingIndexPromise = new Promise(resolve => {
      const script = document.createElement('script'); script.src = 'rankings.js';
      let done = false;
      const finish = value => {
        if (done) return; done = true; clearTimeout(timer); script.remove(); resolve(value);
      };
      const timer = setTimeout(() => finish(null), 30000);
      script.onerror = () => finish(null);
      script.onload = () => {
        const index = window.SPX_RANKINGS;
        finish(index?.version === 1 && index.source === M.rankingSource(C) ? index : null);
      };
      document.head.append(script);
    });
    return rankingIndexPromise;
  }
  function finishLibrarySort() {
    sortBusy = false; $('strategyList').setAttribute('aria-busy','false');
    $('addMatching').disabled = !state.filtered.length;
    renderLibrary();
  }
  async function sortLibrary() {
    const token = ++sortRevision, sort = $('sortBy').value, exposure = $('sortExposure').value;
    sortError = null; $('sortExposure').disabled = !sort;
    if (!sort) {
      $('sortStatus').textContent = 'Sort uses the chart’s date range.';
      finishLibrarySort(); return;
    }
    const start = state.start, end = state.end, key = start + ':' + end + ':' + exposure;
    if (sortWindow !== key) { sortWindow = key; sortScores = new Map(); }
    const scores = sortScores, records = state.filtered.slice();
    const [metric,direction] = sort.split('-');
    sortBusy = true; $('strategyList').setAttribute('aria-busy','true');
    $('addMatching').disabled = true; $('sortStatus').textContent = 'Calculating rankings…'; renderLibrary();
    try {
      if (records.some(r => !scores.has(r.id))) {
        const index = await loadRankingIndex(); if (token !== sortRevision) return;
        const stored = index?.windows?.[start + ':' + end];
        if (Array.isArray(stored) && stored.length === C.strategies.length * 4) {
          for (const r of records) {
            const offset = catalogOrder.get(r.id) * 4 + (exposure === 'spx' ? 2 : 0);
            scores.set(r.id,{cagr:stored[offset],sharpe:stored[offset+1]});
          }
        } else {
          // Custom date windows use the same exact NAV statistics as the chart.
          const groups = new Map();
          for (const r of records) if (!scores.has(r.id)) {
            if (!groups.has(r.chunk)) groups.set(r.chunk,[]);
            groups.get(r.chunk).push(r);
          }
          for (const [chunk,rows] of groups) {
            if (token !== sortRevision) return;
            const buffer = await loadChunk(chunk);
            if (token !== sortRevision) return;
            for (const r of rows) {
              const nav = new Float64Array(buffer,(r.slot*2+(exposure === 'spx' ? 1 : 0))*N*8,N);
              const stats = M.statistics(nav,start,end,C.initial,dates);
              scores.set(r.id,{cagr:stats.cagr,sharpe:stats.sharpe});
            }
            // Keep selected chart series in memory; discard sorting-only series.
            if (![...state.selected.keys()].some(id => byId.get(id).chunk === chunk)) {
              cache.delete(chunk); loading.delete(chunk);
            }
            $('sortStatus').textContent = 'Calculating rankings · ' + count(records.filter(r => scores.has(r.id)).length) + ' / ' + count(records.length);
            await frame();
          }
        }
      }
      if (token !== sortRevision) return;
      state.filtered = records.sort((a,b) => M.compareScores(scores.get(a.id)?.[metric],scores.get(b.id)?.[metric],direction === 'desc') || catalogOrder.get(a.id)-catalogOrder.get(b.id));
      $('strategyList').scrollTop = 0;
      $('sortStatus').textContent = (metric === 'cagr' ? 'CAGR' : 'Sharpe') + ' · ' + modeName(exposure) + ' · Chart dates. Unavailable values last.';
      finishLibrarySort();
    } catch (error) {
      if (token !== sortRevision) return;
      sortError = error.message;
      $('sortStatus').textContent = 'Sorting could not finish. Change a filter or choose the sort again to retry.';
      finishLibrarySort();
    }
  }
  function renderLibrary() {
    const list = $('strategyList'), h = 66, start = Math.max(0, Math.floor(list.scrollTop / h) - 3);
    const end = Math.min(state.filtered.length, start + Math.ceil(list.clientHeight / h) + 7);
    $('strategySpace').style.height = state.filtered.length * h + 'px';
    $('strategyRows').style.transform = 'translateY(' + start * h + 'px)';
    $('strategyRows').innerHTML = state.filtered.slice(start, end).map(r => {
      const chosen = state.selected.has(r.id), metric = $('sortBy').value.split('-')[0];
      const ranked = metric && !sortBusy && !sortError;
      const score = sortScores.get(r.id)?.[metric];
      return '<label class="strategy-row' + (chosen ? ' chosen' : '') + '" title="' + esc(description(r)) + '">' +
        '<input type="checkbox" data-id="' + esc(r.id) + '"' + (chosen ? ' checked' : '') + ' aria-label="' + esc(r.name + ', ' + r.subtitle) + '">' +
        '<span class="row-text"><strong>' + esc(r.name) + '</strong><small>' + esc(r.subtitle) + '</small></span>' +
        (ranked ? '<span class="sort-value"><b>' + (Number.isFinite(score) ? metric === 'cagr' ? pct(score) : fmt(score,2) : '—') + '</b><small>' + (metric === 'cagr' ? 'CAGR' : 'Sharpe') + '</small></span>' :
          '<span class="mini-tag">' + (r.category === 'Both' ? Math.round(r.premium * 100) + '%' : r.category === 'Put buying' ? 'BUY' : 'SELL') + '</span>') + '</label>';
    }).join('') || '<div class="empty-results">No strategies match.<br>Try fewer filters or another search.</div>';
  }
  function metricCell(values, key) {
    return '<div class="metric-cell">' + values.map((value, i) => {
      const v = value.stats?.[key];
      return '<div class="' + (i ? 'second ' : '') + (v < 0 ? 'negative' : '') + '" title="' + modeName(value.mode) + '">' +
        '<span class="sr-only">' + modeName(value.mode) + ': </span>' + (key === 'sharpe' ? fmt(v, 2) : pct(v)) + '</div>';
    }).join('') + '</div>';
  }
  function renderSelected() {
    const list = $('selectedList'), h = 76, start = Math.max(0, Math.floor(list.scrollTop / h) - 2);
    const end = Math.min(selectedIds.length, start + Math.ceil(list.clientHeight / h) + 5);
    $('selectedSpace').style.height = Math.max(selectedIds.length * h, 90) + 'px';
    $('selectedRows').style.transform = 'translateY(' + start * h + 'px)';
    $('selectedRows').innerHTML = selectedIds.slice(start, end).map(id => {
      const r = byId.get(id), s = state.selected.get(id);
      const values = lineModes(s.mode).map(mode => ({mode, stats: metrics.get(id + ':' + mode)?.stats}));
      return '<div class="selected-row' + (s.visible ? '' : ' muted-row') + '" data-id="' + esc(id) + '">' +
        '<div class="selected-name"><input type="checkbox" data-action="visible"' + (s.visible ? ' checked' : '') + ' aria-label="Show ' + esc(r.name) + '">' +
        '<input class="color-input" type="color" data-action="color" value="' + s.color + '" aria-label="Color for ' + esc(r.name) + '">' +
        '<span class="label" title="' + esc(description(r)) + '"><strong>' + esc(r.name) + '</strong><small>' + esc(detail(r)) + '</small></span></div>' +
        '<select class="exposure-select" data-action="mode" aria-label="Exposure for ' + esc(r.name + ', ' + r.subtitle) + '">' +
        modes.map(mode => '<option value="' + mode + '"' + (s.mode === mode ? ' selected' : '') + '>' + (mode === 'both' ? 'Both' : modeName(mode)) + '</option>').join('') + '</select>' +
        ['total','cagr','vol','sharpe','maxDD'].map(key => metricCell(values, key)).join('') +
        '<button class="remove-button" data-action="remove" aria-label="Remove ' + esc(r.name) + '">×</button></div>';
    }).join('') || '<div class="empty-results">Your selected strategies and their performance will appear here.</div>';
  }
  function add(id, mode = $('addMode').value) {
    if (!state.selected.has(id) && byId.has(id)) state.selected.set(id, {mode, color: color(), visible: true});
  }
  function selectionChanged() {
    selectedIds = [...state.selected.keys()];
    $('strategyCount').textContent = count(selectedIds.length) + ' strateg' + (selectedIds.length === 1 ? 'y' : 'ies');
    renderLibrary(); renderSelected(); scheduleRefresh();
  }
  function metricDifference(value, benchmark, key) {
    if (value == null || benchmark == null || !Number.isFinite(value) || !Number.isFinite(benchmark)) return '—';
    const difference = (value - benchmark) * (key === 'sharpe' ? 1 : 100);
    const rounded = Number(difference.toFixed(key === 'sharpe' ? 2 : 1));
    return (rounded > 0 ? '+' : '') + fmt(rounded, key === 'sharpe' ? 2 : 1) + (key === 'sharpe' ? '' : ' pp');
  }
  function updateChartMetrics() {
    const compared = state.curves.some(c => c.mode === 'spx');
    const benchmark = metrics.get('benchmark:options');
    chartMetricRows = state.curves.filter(c => c.id !== 'benchmark');
    if (state.benchmark || compared) chartMetricRows.unshift({id:'benchmark',mode:'options',color:'#47566f',...benchmark});
    $('chartMetrics').hidden = !chartMetricRows.length;
    $('chartMetrics').classList.toggle('narrow', $('chartMetrics').clientWidth < 520);
    chartMetricRowHeight = $('chartMetrics').classList.contains('narrow') ? 72 : 48;
    $('chartMetricsCount').textContent = count(chartMetricRows.length) + (chartMetricRows.length === 1 ? ' portfolio' : ' portfolios');
    $('chartMetricsNote').hidden = !compared;
    $('chartMetricsList').style.height = Math.min(chartMetricRows.length * chartMetricRowHeight, chartMetricRowHeight * 3) + 'px';
    $('chartMetricsSpace').style.height = chartMetricRows.length * chartMetricRowHeight + 'px';
    renderChartMetrics();
  }
  function renderChartMetrics() {
    const list = $('chartMetricsList'), h = chartMetricRowHeight;
    const start = Math.max(0, Math.floor(list.scrollTop / h) - 1);
    const end = Math.min(chartMetricRows.length, start + Math.ceil(list.clientHeight / h) + 3);
    const benchmark = metrics.get('benchmark:options')?.stats;
    $('chartMetricsRows').style.transform = 'translateY(' + start * h + 'px)';
    $('chartMetricsRows').innerHTML = chartMetricRows.slice(start,end).map(c => {
      const r = byId.get(c.id), label = r ? r.name : state.benchmark ? 'SPX benchmark' : 'SPX reference';
      const subtitle = r ? r.subtitle + ' · ' + modeName(c.mode) : 'Price index · Same date range';
      return '<div class="chart-metric-row" data-id="' + esc(c.id) + '" data-mode="' + c.mode + '">' +
        '<div class="chart-metric-name" title="' + esc(label + ' · ' + subtitle) + '"><i style="border-color:' + c.color + ';border-top-style:' + (c.mode === 'spx' ? 'dashed' : 'solid') + '"></i>' +
        '<span><strong>' + esc(label) + '</strong><small>' + esc(subtitle) + '</small></span></div>' +
        chartMetricKeys.map(key => '<div class="chart-metric-value" data-metric="' + key + '"><b>' + (key === 'sharpe' ? fmt(c.stats[key],2) : pct(c.stats[key])) + '</b>' +
          (c.mode === 'spx' ? '<small title="Difference versus SPX for the selected dates">' + metricDifference(c.stats[key],benchmark?.[key],key) + '</small>' : '') + '</div>').join('') + '</div>';
    }).join('');
  }
  function setRange(from, to, preset = '') {
    try {
      [state.start, state.end] = M.windowIndices(dates, from, to);
      $('from').value = dates[state.start]; $('to').value = dates[state.end];
      document.querySelectorAll('[data-period]').forEach(b => b.classList.toggle('active', b.dataset.period === preset));
      syncRange(); scheduleRefresh(); return true;
    } catch (error) { toast(error.message); return false; }
  }
  function syncRange() {
    $('rangeStart').value = state.start; $('rangeEnd').value = state.end;
    $('navSelection').style.left = state.start / (N - 1) * 100 + '%';
    $('navSelection').style.width = (state.end - state.start) / (N - 1) * 100 + '%';
    $('rangeLabel').textContent = friendlyDate(dates[state.start]) + ' — ' + friendlyDate(dates[state.end]);
    $('sessionCount').textContent = count(state.end - state.start + 1) + ' sessions';
    const next = state.start + ':' + state.end;
    if (sortRange !== next) { sortRange = next; if ($('sortBy').value) sortLibrary(); }
  }
  function scheduleRefresh(delay = 0) {
    revision++; state.busy = true; state.error = null;
    $('chartProgress').hidden = false; $('chartProgress').textContent = 'Preparing comparison…';
    $('chartMetrics').hidden = true;
    $('saveImage').disabled = true; $('exportMetrics').disabled = true;
    $('tooltip').hidden = true; clearOverlay();
    clearTimeout(refreshTimer); refreshTimer = setTimeout(() => refresh(revision), delay);
  }
  async function refresh(token) {
    try {
      const records = [...state.selected.entries()], keys = [...new Set(records.map(([id]) => byId.get(id).chunk))];
      const pending = keys.filter(key => !cache.has(key)); let loaded = 0;
      if (pending.length) $('chartProgress').textContent = 'Loading data · 0 / ' + count(pending.length) + ' files';
      await Promise.all(pending.map(key => loadChunk(key).then(() => {
        loaded++; if (token === revision) $('chartProgress').textContent = 'Loading data · ' + count(loaded) + ' / ' + count(pending.length) + ' files';
      })));
      if (token !== revision) return;
      const start = state.start, end = state.end, drawdown = state.view === 'drawdown';
      const nextWindow = start + ':' + end + ':' + drawdown;
      if (metricWindow !== nextWindow) { metrics = new Map(); metricWindow = nextWindow; }
      const wanted = [];
      for (const [id, s] of records) for (const mode of lineModes(s.mode)) wanted.push({id, ...s, mode});
      if (state.benchmark || wanted.some(c => c.visible && c.mode === 'spx'))
        wanted.push({id: 'benchmark', mode: 'options', color: '#47566f', visible: state.benchmark});
      let batch = performance.now();
      for (let i = 0; i < wanted.length; i++) {
        const item = wanted[i], key = item.id + ':' + item.mode;
        if (!metrics.has(key)) {
          const nav = navFor(item.id, item.mode), stats = M.statistics(nav, start, end, C.initial, dates, drawdown);
          metrics.set(key, {nav, stats, start});
        }
        if (performance.now() - batch > 10) {
          $('chartProgress').textContent = 'Calculating · ' + count(i + 1) + ' / ' + count(wanted.length) + ' lines';
          await frame(); if (token !== revision) return; batch = performance.now();
        }
      }
      const keep = new Set(wanted.map(v => v.id + ':' + v.mode));
      for (const key of metrics.keys()) if (!keep.has(key)) metrics.delete(key);
      const curves = wanted.filter(v => v.visible).map(item => ({...item, ...metrics.get(item.id + ':' + item.mode)}));
      state.curves = curves;
      $('lineCount').textContent = count(curves.length) + (curves.length === 1 ? ' line' : ' lines');
      $('chartEmpty').hidden = curves.length > 0;
      renderSelected(); drawNavigator(); await drawChart(curves, token);
      if (token !== revision) return;
      state.busy = false; state.ready = true;
      updateChartMetrics();
      $('chartProgress').hidden = true; $('saveImage').disabled = !curves.length; $('exportMetrics').disabled = !wanted.length;
      $('status').textContent = count(curves.length) + ' visible lines · Ready';
    } catch (error) {
      if (token !== revision) return;
      state.busy = false; state.error = error.message;
      $('chartProgress').textContent = 'Data could not load. Reselect a line to retry.';
      $('status').textContent = error.message; toast(error.message); console.error(error);
    }
  }
  function canvasContext(canvas) {
    const {width, height} = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {ctx, width, height};
  }
  function clearOverlay() {
    const canvas = $('overlay'), ctx = canvas.getContext('2d');
    ctx.save(); ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,canvas.width,canvas.height); ctx.restore();
  }
  function axisLabel(v) { return state.view === 'growth' ? '$' + shortNumber(v) : shortNumber(v) + '%'; }
  function xAt(i) { return plot.left + (i - state.start) / Math.max(1, state.end - state.start) * plot.w; }
  function yAt(v) { return plot.top + (plot.hi - v) / (plot.hi - plot.lo) * plot.h; }
  function indexAt(x) { return Math.max(state.start, Math.min(state.end, Math.round(state.start + (x - plot.left) / plot.w * (state.end - state.start)))); }
  async function drawChart(curves, token) {
    const {ctx, width, height} = canvasContext($('chart')); canvasContext($('overlay')); hoverIndex = null;
    const left = width < 500 ? 56 : 66, right = width < 600 ? 16 : 70, top = 25, bottom = 35;
    let lo = state.view === 'growth' ? 100 : 0, hi = lo;
    for (const c of curves) {
      lo = Math.min(lo, state.view === 'drawdown' ? c.stats.maxDD * 100 : c.stats.min - (state.view === 'return' ? 100 : 0));
      hi = Math.max(hi, state.view === 'drawdown' ? 0 : c.stats.max - (state.view === 'return' ? 100 : 0));
    }
    const span = hi - lo || (state.view === 'growth' ? 20 : 10); lo -= span * .07; hi += span * .09;
    plot = {left, top, w: width - left - right, h: height - top - bottom, lo, hi, width, height};
    ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,width,height);
    ctx.font = '10px "Segoe UI", sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const tick of M.niceTicks(lo,hi,5).ticks) {
      const y = yAt(tick); ctx.beginPath(); ctx.moveTo(left,y); ctx.lineTo(left+plot.w,y);
      ctx.strokeStyle = '#eaf0f6'; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = '#8c9bb0'; ctx.fillText(axisLabel(tick),left-10,y);
    }
    const tickCount = Math.max(2, Math.min(7, Math.floor(plot.w/110))), used = new Set();
    for (let n = 0; n <= tickCount; n++) {
      const i = Math.round(state.start + (state.end-state.start)*n/tickCount); if (used.has(i)) continue; used.add(i);
      const d = new Date(M.time(dates[i]));
      const label = d.toLocaleDateString('en-US', {month:'short', ...(state.end-state.start > 180 ? {year:'2-digit'} : {day:'numeric'}), timeZone:'UTC'});
      ctx.textAlign = n === 0 ? 'left' : n === tickCount ? 'right' : 'center';
      ctx.fillStyle = '#92a0b3'; ctx.fillText(label,xAt(i),height-13);
    }
    const baseline = state.view === 'growth' ? 100 : 0;
    ctx.beginPath(); ctx.moveTo(left,yAt(baseline)); ctx.lineTo(left+plot.w,yAt(baseline));
    ctx.strokeStyle = '#bfccdf'; ctx.setLineDash([3,4]); ctx.stroke(); ctx.setLineDash([]);
    ctx.save(); ctx.beginPath(); ctx.rect(left-1,top-1,plot.w+2,plot.h+2); ctx.clip();
    let batch = performance.now();
    for (let i = 0; i < curves.length; i++) {
      drawCurve(ctx, curves[i], curves.length > 200 ? .16 : curves.length > 30 ? .42 : .92);
      if (performance.now()-batch > 12) {
        $('chartProgress').textContent = 'Drawing · ' + count(i+1) + ' / ' + count(curves.length) + ' lines';
        await frame(); if (token !== revision) return; batch = performance.now();
      }
    }
    ctx.restore();
    if (curves.length <= 10 && width >= 600) {
      const labels = curves.map(c => ({c, y:yAt(M.valueAt(c,state.end,state.view))})).sort((a,b) => a.y-b.y);
      let previous = top-15;
      for (const label of labels) { label.y = Math.max(label.y, previous+15); previous = label.y; }
      if (labels.length && labels.at(-1).y > top+plot.h) {
        labels.at(-1).y = top+plot.h;
        for (let i = labels.length-2; i >= 0; i--) labels[i].y = Math.min(labels[i].y, labels[i+1].y-15);
      }
      ctx.font = '600 10px "Segoe UI", sans-serif'; ctx.textAlign = 'left';
      for (const {c,y} of labels) { ctx.fillStyle = c.color; ctx.fillText(axisLabel(M.valueAt(c,state.end,state.view)),left+plot.w+9,y); }
    }
  }
  function drawCurve(ctx, c, opacity = 1, highlight = false) {
    ctx.strokeStyle = c.color; ctx.globalAlpha = opacity; ctx.lineWidth = highlight ? 2.6 : c.id === 'benchmark' ? 1.7 : 1.5;
    ctx.setLineDash(c.mode === 'spx' ? [6,4] : c.id === 'benchmark' ? [2,3] : []); ctx.beginPath();
    const length = state.end - state.start + 1;
    if (length <= plot.w * 2) {
      for (let i = state.start; i <= state.end; i++) {
        const x = xAt(i), y = yAt(M.valueAt(c,i,state.view));
        if (i === state.start) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
    } else {
      // Preserve each pixel column's extrema, including short-lived losses.
      let begun = false; const buckets = Math.max(1, Math.floor(plot.w));
      for (let b = 0; b < buckets; b++) {
        const first = state.start + Math.floor(b*length/buckets), last = state.start + Math.floor((b+1)*length/buckets);
        let low = Infinity, high = -Infinity, lowI = first, highI = first;
        for (let i = first; i < last; i++) {
          const v = M.valueAt(c,i,state.view);
          if (v < low) { low = v; lowI = i; } if (v > high) { high = v; highI = i; }
        }
        for (const i of lowI < highI ? [lowI,highI] : [highI,lowI]) {
          const x = xAt(i), y = yAt(M.valueAt(c,i,state.view));
          if (!begun) {ctx.moveTo(x,y); begun = true;} else ctx.lineTo(x,y);
        }
      }
      ctx.lineTo(xAt(state.end), yAt(M.valueAt(c,state.end,state.view)));
    }
    ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    if (state.start === state.end) {
      ctx.beginPath(); ctx.arc(xAt(state.start),yAt(M.valueAt(c,state.start,state.view)),3,0,Math.PI*2); ctx.fillStyle = c.color; ctx.fill();
    }
  }
  function drawNavigator() {
    const {ctx, width, height} = canvasContext($('navChart'));
    const lo = Math.min(...C.spx), hi = Math.max(...C.spx), y = v => height-4-(v-lo)/(hi-lo)*(height-9);
    ctx.beginPath(); ctx.moveTo(0,height); C.spx.forEach((v,i) => ctx.lineTo(i/(N-1)*width,y(v)));
    ctx.lineTo(width,height); ctx.closePath(); ctx.fillStyle = '#e9eef6'; ctx.fill();
    ctx.beginPath(); C.spx.forEach((v,i) => i ? ctx.lineTo(i/(N-1)*width,y(v)) : ctx.moveTo(0,y(v)));
    ctx.lineWidth = 1; ctx.strokeStyle = '#a9b9d1'; ctx.stroke(); syncRange();
  }
  function showHover(x, y) {
    if (!plot || state.busy || !state.curves.length) return; clearOverlay();
    if (x < plot.left || x > plot.left+plot.w || y < plot.top || y > plot.top+plot.h) { $('tooltip').hidden = true; return; }
    const i = indexAt(x); hoverIndex = i; const ctx = $('overlay').getContext('2d'), xx = xAt(i);
    ctx.strokeStyle = '#a9b8ce'; ctx.setLineDash([3,3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(xx,plot.top); ctx.lineTo(xx,plot.top+plot.h); ctx.stroke(); ctx.setLineDash([]);
    const closest = [];
    for (const c of state.curves) {
      const value = M.valueAt(c,i,state.view), yy = yAt(value), distance = Math.abs(yy-y);
      if (closest.length < 8 || distance < closest.at(-1).distance) {
        closest.push({c,value,yy,distance}); closest.sort((a,b) => a.distance-b.distance); if (closest.length > 8) closest.pop();
      }
    }
    if (state.curves.length <= 8) closest.sort((a,b) => state.curves.indexOf(a.c)-state.curves.indexOf(b.c));
    for (const item of closest) { ctx.beginPath(); ctx.arc(xx,item.yy,3,0,Math.PI*2); ctx.fillStyle = item.c.color; ctx.fill(); }
    $('tooltip').innerHTML = '<div class="tooltip-date">' + friendlyDate(dates[i]) + '</div>' + closest.map(({c,value}) => {
      const r = byId.get(c.id), label = r ? r.name + ' · ' + r.subtitle + ' · ' + modeName(c.mode) : 'SPX benchmark';
      return '<div class="tooltip-line"><i style="border-color:' + c.color + ';border-top-style:' + (c.mode === 'spx' ? 'dashed' : 'solid') + '"></i><span>' + esc(label) + '</span><b>' +
        (state.view === 'growth' ? '$' + fmt(value) : fmt(value) + '%') + '</b></div>';
    }).join('') + (state.curves.length > 8 ? '<div class="tooltip-note">8 nearest lines of ' + count(state.curves.length) + ' · Move vertically to explore</div>' : '');
    const tip = $('tooltip'); tip.hidden = false;
    tip.style.left = Math.max(5,Math.min(x+18,plot.width-tip.offsetWidth-5)) + 'px';
    tip.style.top = Math.max(5,Math.min(y+12,plot.height-tip.offsetHeight-5)) + 'px';
  }
  function highlightRow(id) {
    if (state.busy || !plot) return; clearOverlay(); $('tooltip').hidden = true;
    const ctx = $('overlay').getContext('2d'); ctx.save(); ctx.beginPath(); ctx.rect(plot.left,plot.top,plot.w,plot.h); ctx.clip();
    for (const c of state.curves) if (c.id === id) drawCurve(ctx,c,1,true); ctx.restore();
  }
  function setup() {
    return {format:'spx-research-comparison', version:1, from:dates[state.start], to:dates[state.end], view:state.view,
      benchmark:state.benchmark, selected:[...state.selected].map(([id,s]) => ({id,...s})),
      filters:{category:state.category, search:$('search').value, addMode:$('addMode').value, sortBy:$('sortBy').value, sortExposure:$('sortExposure').value, ...Object.fromEntries(filterIds.map(id => [id,$(id).value]))}};
  }
  function applySetup(value) {
    if (value?.format !== 'spx-research-comparison' || value.version !== 1 || !Array.isArray(value.selected)) throw Error('This is not an SPX comparison setup.');
    const [start,end] = M.windowIndices(dates,value.from,value.to);
    if (!['growth','return','drawdown'].includes(value.view)) throw Error('Unknown chart view in setup.');
    const selected = new Map(); let skipped = 0;
    for (const item of value.selected) {
      if (!byId.has(item.id)) { skipped++; continue; }
      if (!modes.includes(item.mode) || !/^#[a-f0-9]{6}$/i.test(item.color) || typeof item.visible !== 'boolean') throw Error('Invalid line settings in setup.');
      selected.set(item.id,{mode:item.mode,color:item.color,visible:item.visible});
    }
    state.selected = selected; colorIndex = selected.size; state.start = start; state.end = end;
    state.view = value.view; state.benchmark = value.benchmark === true;
    $('view').value = state.view; $('benchmark').checked = state.benchmark;
    const filters = value.filters || {}; state.category = ['','Put selling','Put buying','Both'].includes(filters.category) ? filters.category : '';
    $('search').value = typeof filters.search === 'string' ? filters.search : '';
    for (const id of [...filterIds,'addMode']) $(id).value = [...$(id).options].some(o => o.value === filters[id]) ? filters[id] : (id === 'addMode' ? 'options' : '');
    for (const id of ['sortBy','sortExposure']) $(id).value = [...$(id).options].some(o => o.value === filters[id]) ? filters[id] : (id === 'sortExposure' ? 'options' : '');
    document.querySelectorAll('[data-category]').forEach(b => b.setAttribute('aria-pressed',String(b.dataset.category === state.category)));
    filter(); setRange(dates[start],dates[end],start === 0 && end === N-1 ? 'ALL' : ''); selectionChanged(); return skipped;
  }
  function download(blob, name) {
    const link = document.createElement('a'), url = URL.createObjectURL(blob);
    link.href = url; link.download = name; document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url),10000);
  }
  function metricsCSV() {
    if (state.busy) throw Error('Wait for the current comparison to finish loading.');
    const rows = [['strategy_id','strategy','details','exposure','visible','first_session','last_session','observations','total_return','cagr','annualized_volatility','sharpe_zero_cash_rate','max_drawdown']];
    const append = (id,mode,s) => {
      const c = metrics.get(id+':'+mode), r = byId.get(id); if (!c) return;
      rows.push([id,r?.name || 'SPX benchmark',r ? detail(r) : 'SPX price index',id === 'benchmark' ? 'SPX' : modeName(mode),s.visible,
        dates[state.start],dates[state.end],c.stats.observations,...['total','cagr','vol','sharpe','maxDD'].map(k => c.stats[k])]);
    };
    for (const [id,s] of state.selected) for (const mode of lineModes(s.mode)) append(id,mode,s);
    if (state.benchmark) append('benchmark','options',{visible:true});
    return '\uFEFF' + rows.map(row => row.map(v => '"'+String(v ?? '').replaceAll('"','""')+'"').join(',')).join('\r\n');
  }
  function saveImage() {
    if (state.busy) return;
    const width = Math.max(1000,Math.round(plot.width)), scale = width/plot.width;
    const lines = state.curves.slice(0,18), legendH = Math.ceil(lines.length/2)*24 + 25;
    const canvas = document.createElement('canvas'); canvas.width = width*2; canvas.height = Math.ceil(100+plot.height*scale+legendH)*2;
    const ctx = canvas.getContext('2d'); ctx.scale(2,2); ctx.fillStyle = '#fff'; ctx.fillRect(0,0,canvas.width/2,canvas.height/2);
    ctx.fillStyle = '#182b46'; ctx.font = '600 24px "Segoe UI",sans-serif'; ctx.fillText('SPX Research / Portfolio performance',28,38);
    ctx.fillStyle = '#748197'; ctx.font = '13px "Segoe UI",sans-serif';
    ctx.fillText(friendlyDate(dates[state.start])+' — '+friendlyDate(dates[state.end])+' · '+$('view').selectedOptions[0].textContent,28,64);
    ctx.drawImage($('chart'),0,88,width,plot.height*scale);
    drawExportMetrics(ctx,scale,88);
    lines.forEach((c,i) => {
      const r = byId.get(c.id), x = 28+(i%2)*(width/2), y = 104+plot.height*scale+Math.floor(i/2)*24;
      ctx.strokeStyle = c.color; ctx.lineWidth = 2; ctx.setLineDash(c.mode === 'spx' ? [6,4] : []);
      ctx.beginPath(); ctx.moveTo(x,y-4); ctx.lineTo(x+20,y-4); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#52647e'; ctx.font = '11px "Segoe UI",sans-serif';
      ctx.fillText(r ? r.name+' · '+r.subtitle+' · '+modeName(c.mode) : 'SPX benchmark',x+29,y,width/2-72);
    });
    if (state.curves.length > 18) { ctx.fillStyle = '#748197'; ctx.fillText('+'+count(state.curves.length-18)+' more lines; export metrics for the complete selection.',28,canvas.height/2-9); }
    canvas.toBlob(blob => { if (blob) download(blob,'SPX-comparison-'+dates[state.start]+'-to-'+dates[state.end]+'.png'); });
  }
  function drawExportMetrics(ctx,scale,top) {
    const panel = $('chartMetrics'); if (panel.hidden) return;
    const rect = panel.getBoundingClientRect(), wrap = $('chartWrap').getBoundingClientRect();
    const narrow = panel.classList.contains('narrow'), width = rect.width, height = rect.height;
    const list = $('chartMetricsList'), listRect = list.getBoundingClientRect(), rowHeight = chartMetricRowHeight;
    const bodyTop = listRect.top-rect.top, start = Math.floor(list.scrollTop/rowHeight);
    const benchmark = metrics.get('benchmark:options')?.stats;
    const columnX = i => narrow ? 9+(width-18)*(i+.5)/4 : width-11-(3-i)*81;
    ctx.save(); ctx.translate((rect.left-wrap.left)*scale,top+(rect.top-wrap.top)*scale); ctx.scale(scale,scale);
    ctx.fillStyle = '#fffffff5'; ctx.fillRect(0,0,width,height); ctx.strokeStyle = '#dfe6f0'; ctx.lineWidth = 1; ctx.strokeRect(.5,.5,width-1,height-1);
    ctx.fillStyle = '#182b46'; ctx.font = '600 10px "Segoe UI",sans-serif'; ctx.textAlign = 'left'; ctx.fillText('Period performance',11,18);
    ctx.font = '8px "Segoe UI",sans-serif'; ctx.textAlign = 'right'; ctx.fillStyle = '#8190a6'; ctx.fillText(count(chartMetricRows.length)+' portfolios',width-11,18);
    if (!narrow) {ctx.textAlign='left';ctx.fillText('Portfolio',11,bodyTop-8);}
    ctx.textAlign = narrow ? 'center' : 'right';
    ['CAGR','Ann. vol.','Max drawdown','Sharpe'].forEach((label,i) => ctx.fillText(label,columnX(i),bodyTop-8));
    ctx.save(); ctx.beginPath(); ctx.rect(0,bodyTop,width,list.clientHeight); ctx.clip();
    for(let j=start;j<Math.min(chartMetricRows.length,start+Math.ceil(list.clientHeight/rowHeight)+1);j++) {
      const c=chartMetricRows[j],r=byId.get(c.id),y=bodyTop+j*rowHeight-list.scrollTop;
      const label=r?r.name:'SPX benchmark', subtitle=r?r.subtitle+' · '+modeName(c.mode):'Price index · Same date range';
      const nameWidth=narrow?width-39:width-353;
      ctx.strokeStyle='#edf1f6';ctx.beginPath();ctx.moveTo(0,y+rowHeight);ctx.lineTo(width,y+rowHeight);ctx.stroke();
      ctx.strokeStyle=c.color;ctx.lineWidth=2;ctx.setLineDash(c.mode==='spx'?[4,3]:[]);ctx.beginPath();ctx.moveTo(11,y+14);ctx.lineTo(23,y+14);ctx.stroke();ctx.setLineDash([]);ctx.lineWidth=1;
      ctx.fillStyle='#182b46';ctx.textAlign='left';ctx.font='600 9px "Segoe UI",sans-serif';ctx.fillText(label,29,y+17,nameWidth);
      ctx.fillStyle='#8592a6';ctx.font='8px "Segoe UI",sans-serif';ctx.fillText(subtitle,29,y+29,nameWidth);
      chartMetricKeys.forEach((key,i)=>{
        ctx.textAlign=narrow?'center':'right';ctx.fillStyle='#182b46';ctx.font='600 10px "Segoe UI",sans-serif';
        ctx.fillText(key==='sharpe'?fmt(c.stats[key],2):pct(c.stats[key]),columnX(i),y+(narrow?47:21));
        if(c.mode==='spx'){ctx.fillStyle='#8392a8';ctx.font='8px "Segoe UI",sans-serif';ctx.fillText(metricDifference(c.stats[key],benchmark?.[key],key),columnX(i),y+(narrow?60:34));}
      });
    }
    ctx.restore();
    if(!$('chartMetricsNote').hidden){ctx.textAlign='left';ctx.fillStyle='#7c8ca3';ctx.font='8px "Segoe UI",sans-serif';ctx.fillText('With SPX: differences vs SPX · pp = percentage points',10,height-7,width-20);}
    ctx.restore();
  }
  function example() {
    state.selected.clear(); colorIndex = 0;
    for (const [category,hedge] of [['Put selling',null],['Both','Downside buffer'],['Both','Long put']]) {
      const r = C.strategies.find(r => r.category === category && r.hedge === hedge && r.variation === '98/95' && r.days === 7 && (category !== 'Both' || r.premium === .2));
      if (r) add(r.id,'both');
    }
    state.benchmark = true; $('benchmark').checked = true; selectionChanged();
  }
  // Delegated events keep the DOM small even with the entire library selected.
  $('strategyRows').addEventListener('change',e => {
    const id = e.target.dataset.id; if (!id) return;
    if (e.target.checked) add(id); else state.selected.delete(id); selectionChanged();
  });
  $('strategyList').addEventListener('scroll',renderLibrary,{passive:true});
  $('selectedList').addEventListener('scroll',renderSelected,{passive:true});
  $('chartMetricsList').addEventListener('scroll',renderChartMetrics,{passive:true});
  $('chartMetricsRows').addEventListener('pointerover',e => {
    const id = e.target.closest('[data-id]')?.dataset.id; if (id) highlightRow(id);
  });
  $('chartMetricsRows').addEventListener('pointerleave',clearOverlay);
  $('selectedRows').addEventListener('change',e => {
    const row = e.target.closest('[data-id]'); if (!row) return; const s = state.selected.get(row.dataset.id);
    if (e.target.dataset.action === 'mode') s.mode = e.target.value;
    if (e.target.dataset.action === 'visible') s.visible = e.target.checked;
    if (e.target.dataset.action === 'color') s.color = e.target.value;
    selectionChanged();
  });
  $('selectedRows').addEventListener('click',e => {
    if (e.target.dataset.action === 'remove') { state.selected.delete(e.target.closest('[data-id]').dataset.id); selectionChanged(); }
  });
  $('selectedRows').addEventListener('pointerover',e => {
    const id = e.target.closest('[data-id]')?.dataset.id;
    if (id && id !== hoverRow) { hoverRow = id; highlightRow(id); }
  });
  $('selectedRows').addEventListener('pointerleave',() => { hoverRow = null; clearOverlay(); });
  let searchTimer;
  $('search').addEventListener('input',() => { clearTimeout(searchTimer); searchTimer = setTimeout(filter,100); });
  filterIds.forEach(id => $(id).addEventListener('change',filter));
  for (const id of ['sortBy','sortExposure']) $(id).addEventListener('change',filter);
  $('categories').addEventListener('click',e => {
    if (!e.target.hasAttribute('data-category')) return; state.category = e.target.dataset.category;
    if (state.category && state.category !== 'Both') { $('premium').value = ''; $('hedge').value = ''; }
    document.querySelectorAll('[data-category]').forEach(b => b.setAttribute('aria-pressed',String(b === e.target))); filter();
  });
  $('resetFilters').addEventListener('click',() => {
    state.category = ''; $('search').value = ''; filterIds.forEach(id => $(id).value = '');
    $('sortBy').value = ''; $('sortExposure').value = 'options';
    document.querySelectorAll('[data-category]').forEach(b => b.setAttribute('aria-pressed',String(!b.dataset.category))); filter();
  });
  $('addMatching').addEventListener('click',() => {
    const before = state.selected.size; state.filtered.forEach(r => add(r.id)); selectionChanged(); toast(count(state.selected.size-before)+' strategies added');
  });
  document.querySelectorAll('[data-all-mode]').forEach(b => b.addEventListener('click',() => {
    for (const s of state.selected.values()) s.mode = b.dataset.allMode; selectionChanged();
  }));
  $('clear').addEventListener('click',() => { state.selected.clear(); colorIndex = 0; selectionChanged(); });
  $('benchmark').addEventListener('change',() => { state.benchmark = $('benchmark').checked; scheduleRefresh(); });
  $('view').addEventListener('change',() => { state.view = $('view').value; scheduleRefresh(); });
  $('presets').addEventListener('click',e => {
    const period = e.target.dataset.period; if (period) setRange(...M.presetDates(dates,period),period);
  });
  $('applyDates').addEventListener('click',() => setRange($('from').value,$('to').value));
  for (const id of ['from','to']) $(id).addEventListener('keydown',e => { if (e.key === 'Enter') setRange($('from').value,$('to').value); });
  for (const id of ['rangeStart','rangeEnd']) {
    $(id).max = N-1;
    $(id).addEventListener('input',() => {
      const start = id === 'rangeStart' ? Math.min(Number($(id).value),state.end) : state.start;
      const end = id === 'rangeEnd' ? Math.max(Number($(id).value),state.start) : state.end;
      setRange(dates[start],dates[end]); scheduleRefresh(100);
    });
  }
  const overlay = $('overlay'); let moveFrame = 0;
  overlay.addEventListener('pointermove',e => {
    const rect = overlay.getBoundingClientRect(), x = e.clientX-rect.left, y = e.clientY-rect.top;
    cancelAnimationFrame(moveFrame); moveFrame = requestAnimationFrame(() => {
      if (drag) {
        drag.x = Math.max(plot.left,Math.min(plot.left+plot.w,x)); clearOverlay();
        const ctx = overlay.getContext('2d'); ctx.fillStyle = '#3979e72a';
        ctx.fillRect(Math.min(drag.start,drag.x),plot.top,Math.abs(drag.x-drag.start),plot.h);
      } else showHover(x,y);
    });
  });
  overlay.addEventListener('pointerdown',e => {
    if (e.button !== 0 || !plot || state.busy) return;
    const rect = overlay.getBoundingClientRect(), x = e.clientX-rect.left, y = e.clientY-rect.top;
    if (x < plot.left || x > plot.left+plot.w || y < plot.top || y > plot.top+plot.h) return;
    drag = {start:x,x}; $('tooltip').hidden = true; overlay.setPointerCapture(e.pointerId);
  });
  overlay.addEventListener('pointerup',e => {
    if (!drag) return; const d = drag; drag = null; cancelAnimationFrame(moveFrame); clearOverlay();
    if (overlay.hasPointerCapture(e.pointerId)) overlay.releasePointerCapture(e.pointerId);
    const x = Math.max(plot.left,Math.min(plot.left+plot.w,e.clientX-overlay.getBoundingClientRect().left));
    if (Math.abs(x-d.start) > 6) setRange(dates[indexAt(Math.min(d.start,x))],dates[indexAt(Math.max(d.start,x))]);
    else showHover(x,e.clientY-overlay.getBoundingClientRect().top);
  });
  overlay.addEventListener('pointercancel',() => { drag = null; cancelAnimationFrame(moveFrame); clearOverlay(); });
  overlay.addEventListener('pointerleave',() => { if (!drag) { cancelAnimationFrame(moveFrame); clearOverlay(); $('tooltip').hidden = true; } });
  overlay.addEventListener('dblclick',() => setRange(dates[0],dates[N-1],'ALL'));
  overlay.addEventListener('keydown',e => {
    if (!plot || state.busy) return;
    if (e.key === 'Escape') {clearOverlay(); $('tooltip').hidden = true;}
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      e.preventDefault(); const i = Math.max(state.start,Math.min(state.end,(hoverIndex ?? state.end)+(e.key === 'ArrowRight' ? 1 : -1)));
      showHover(xAt(i),plot.top+plot.h/2);
    }
  });
  $('example').addEventListener('click',example);
  for (const id of ['methodBtn','assumptionsBtn']) $(id).addEventListener('click',() => $('methods').showModal());
  $('methods').addEventListener('click',e => {
    if (e.target === $('methods')) { const r=e.target.getBoundingClientRect(); if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close(); }
  });
  $('saveSetup').addEventListener('click',() => download(new Blob([JSON.stringify(setup(),null,2)],{type:'application/json'}),'SPX-comparison-setup.json'));
  $('loadSetup').addEventListener('click',() => $('setupFile').click());
  $('setupFile').addEventListener('change',async () => {
    const file = $('setupFile').files[0]; if (!file) return;
    try {
      if (file.size > 10*1024*1024) throw Error('This setup file is too large. Choose an exported comparison JSON.');
      const skipped = applySetup(JSON.parse(await file.text())); toast('Comparison loaded'+(skipped ? '; '+skipped+' unknown strategies skipped' : ''));
    } catch (error) { toast(error.message); }
    $('setupFile').value = '';
  });
  $('exportMetrics').addEventListener('click',() => download(new Blob([metricsCSV()],{type:'text/csv;charset=utf-8'}),'SPX-metrics-'+dates[state.start]+'-to-'+dates[state.end]+'.csv'));
  $('saveImage').addEventListener('click',saveImage);
  let resizeTimer;
  new ResizeObserver(() => {
    clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { renderLibrary(); renderSelected(); scheduleRefresh(); },120);
  }).observe($('chartWrap'));
  [...new Map(C.strategies.map(r => [r.days,r.tenor])).entries()].sort((a,b) => a[0]-b[0]).forEach(([,tenor]) => $('tenor').add(new Option(tenor,tenor)));
  [...new Set(C.strategies.map(r => r.primary))].sort((a,b) => b-a).forEach(v => $('primary').add(new Option(v+'% of SPX',String(v))));
  [...new Set(C.strategies.map(r => r.width))].sort((a,b) => a-b).forEach(v => $('width').add(new Option(v ? v+' points' : 'Single put',String(v))));
  for (const id of ['from','to']) { $(id).min = dates[0]; $(id).max = dates[N-1]; }
  $('from').value = dates[0]; $('to').value = dates[N-1]; $('benchmark').checked = true;
  $('dataThrough').textContent = 'DATA THROUGH '+friendlyDate(dates[N-1]).toUpperCase();
  $('navChart').parentElement.title = 'SPX across the full research period. Drag either handle to change dates.';
  filter(); syncRange();
  // Every open starts with SPX alone. Saved setups are loaded only on request.
  selectionChanged();
  window.__SPX_APP__ = {
    get ready() { return state.ready; }, get busy() { return state.busy; }, get error() { return state.error; },
    get selectedCount() { return state.selected.size; }, get lineCount() { return state.curves.length; },
    get loadedChunks() { return cache.size; }, get range() { return [state.start,state.end]; },
    get matchedCount() { return state.filtered.length; }, get plot() { return plot; }, setup, applySetup, metricsCSV,
    get sorting() { return {busy:sortBusy,error:sortError,sort:$('sortBy').value,exposure:$('sortExposure').value}; },
    rankedIds: () => state.filtered.map(r => r.id), rank: id => sortScores.get(id),
    statistics: (id,mode='options') => metrics.get(id+':'+mode)?.stats,
    values: (id,mode='options') => navFor(id,mode)
  };
})();
