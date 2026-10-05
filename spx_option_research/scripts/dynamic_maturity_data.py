"""Entry-only features and separately stored future paths for dynamic selection."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from maturity_grid_data import TENORS, SHORTS, WIDTHS, RATIOS
from maturity_grid_engine import trade_paths
from organized_spx_data import DATA_ROOT, SPXSurfaceArchive, load_cash


@dataclass
class DynamicData:
    dates: pd.DatetimeIndex
    cash: pd.Series
    masks: np.ndarray
    entries: np.ndarray
    catalog: pd.DataFrame
    features: dict
    market: pd.DataFrame
    paths: dict
    cycles: dict
    audit: dict
    start: int


def market_features(cash, entry_dates):
    """Only observations strictly before the entry session inform RV/drift."""
    daily = cash.pct_change(fill_method=None)
    result = pd.DataFrame(index=cash.index)
    for window in (21, 63, 126):
        result[f'rv{window}'] = daily.rolling(window, min_periods=window).std(ddof=1).shift(1)*np.sqrt(252)
    result['drift252'] = (daily.rolling(252, min_periods=252).mean().shift(1)*252).clip(-.10, .10)
    return result.reindex(entry_dates)


def build(prepared):
    dates, cash, masks, cycles, source_audit, _, _ = prepared
    started = time.monotonic()
    entries = np.flatnonzero(masks.any(axis=1))
    row_for = {int(entry): i for i, entry in enumerate(entries)}
    catalog = pd.DataFrame([dict(target_dte=t, short_target=s, width=w)
        for t in TENORS for s in SHORTS for w in WIDTHS])
    catalog['long_target'] = catalog.short_target-catalog.width
    geometries = len(SHORTS)*len(WIDTHS)
    shape = (len(entries), len(catalog))
    fields = ('credit', 'risk', 'short_strike', 'long_strike', 'dte', 'closing_cost', 'short_iv', 'long_iv')
    features = {name: np.full(shape, np.nan) for name in fields}
    features['valid'] = np.zeros(shape, dtype=bool)
    features['observed_entry'] = np.zeros(shape, dtype=bool)
    paths = {}
    cycle_map = {}
    all_high = np.repeat(SHORTS,len(WIDTHS))
    all_width = np.tile(WIDTHS,len(SHORTS))
    for tenor_index, tenor in enumerate(TENORS):
        block = slice(tenor_index*geometries, (tenor_index+1)*geometries)
        for cycle in cycles[tenor]:
            row = row_for[cycle.entry]
            path = trade_paths(cycle, SHORTS, WIDTHS, np.array([.25]))
            # Future increments are rebuilt only for actually chosen geometries.
            # Caching every alternative's entire future would waste ~600 MB.
            path.pop('increments')
            path['high'], path['width'] = all_high, all_width
            for field in ('offsets','unresolved','estimated','bound_flags'):
                path[field] = path[field].astype(np.int16)
            paths[(row, tenor_index)] = path
            cycle_map[(row, tenor_index)] = cycle
            hi = np.rint((path['high']-RATIOS[0])*2).astype(int)
            lo = np.rint((path['high']-path['width']-RATIOS[0])*2).astype(int)
            closing_cost = (cycle.cost[0, hi]+cycle.cost[0, lo])*100
            # Avoid quoting a 25%-profit strategy whose present closing friction
            # alone consumes over half that profit target.
            liquid = closing_cost <= .125*path['credit']
            features['observed_entry'][row, block] = path['entry_valid']
            features['valid'][row, block] = path['entry_valid'] & liquid
            for field, value in [('credit',path['credit']), ('risk',path['max_loss']),
                    ('short_strike',path['short_strike']), ('long_strike',path['long_strike']),
                    ('dte',cycle.dte), ('closing_cost',closing_cost)]:
                features[field][row, block] = value
            for key,field in [('credit','credit'),('max_loss','risk'),('short_strike','short_strike'),('long_strike','long_strike')]:
                path[key] = features[field][row,block]
        print(f'Dynamic paths ready: {tenor} DTE ({time.monotonic()-started:.0f}s)', flush=True)
    archive = SPXSurfaceArchive(DATA_ROOT, cache_size=1)
    fingerprints = {}
    for row, entry in enumerate(entries):
        day = dates[entry]
        raw = archive.path_for(day).read_bytes()
        fingerprints[str(day.date())] = hashlib.sha256(raw).hexdigest()
        quotes = pq.read_table(pa.BufferReader(raw), columns=['option_symbol','implied_volatility']).to_pandas()
        quotes['option_symbol'] = quotes.option_symbol.astype(str).str.strip()
        quotes = quotes.loc[~quotes.option_symbol.duplicated(keep=False)].set_index('option_symbol')
        for ti in range(len(TENORS)):
            cycle = cycle_map.get((row, ti))
            if cycle is None:
                continue
            iv = quotes.implied_volatility.reindex(cycle.symbols).to_numpy(float)
            iv = np.where((iv>0)&(iv<5), iv, np.nan)
            high = np.repeat(SHORTS, len(WIDTHS))
            width = np.tile(WIDTHS, len(SHORTS))
            hi = np.rint((high-RATIOS[0])*2).astype(int)
            lo = np.rint((high-width-RATIOS[0])*2).astype(int)
            block = slice(ti*geometries, (ti+1)*geometries)
            features['short_iv'][row, block] = iv[hi]
            features['long_iv'][row, block] = iv[lo]
        if (row+1)%100 == 0:
            print(f'Entry IV features: {row+1}/{len(entries)} ({time.monotonic()-started:.0f}s)', flush=True)
    market = market_features(load_cash(), dates[entries])
    # Identical start for all policies, after 104 weekly opportunities of history.
    eligible_starts = np.flatnonzero((np.arange(len(entries))>=104)&market[['rv21','rv63','rv126']].notna().all(axis=1).to_numpy())
    if not len(eligible_starts):
        raise ValueError('Insufficient trailing history for the common evaluation window')
    start = int(entries[eligible_starts[0]])
    selection_end = start+int((len(dates)-start)*.8)-1
    for key, path in paths.items():
        cycle = cycle_map[key]
        if cycle.entry > selection_end:
            path['selection_unresolved'] = np.zeros(geometries, dtype=np.int32)
            continue
        hi = np.rint((path['high']-RATIOS[0])*2).astype(int)
        lo = np.rint((path['high']-path['width']-RATIOS[0])*2).astype(int)
        end = min(selection_end-cycle.entry, len(cycle.mid)-1)
        raw = cycle.mid[:end+1,hi]-cycle.mid[:end+1,lo]
        friction = cycle.cost[:end+1,hi]+cycle.cost[:end+1,lo]
        half_spread = 2*np.maximum(friction-.03,0)
        feasible = np.isfinite(raw)&(raw+half_spread>=-.011)&(raw-half_spread<=path['actual_width'][None,:]+.011)
        bad = ~feasible|(cycle.quality[:end+1,hi]==0)|(cycle.quality[:end+1,lo]==0)
        offsets = np.minimum(path['offsets'][:,0],end)
        path['selection_unresolved'] = np.cumsum(bad,axis=0)[offsets,np.arange(geometries)]
    audit = dict(**source_audit, candidate_slots=int(np.prod(shape)),
        valid_entry_candidates=int(features['valid'].sum()),
        observed_entry_candidates=int(features['observed_entry'].sum()),
        evaluation_start=str(dates[start].date()),
        entry_iv_fingerprint=hashlib.sha256(''.join(k+v for k,v in sorted(fingerprints.items())).encode()).hexdigest(),
        dynamic_preparation_seconds=time.monotonic()-started)
    print(f'Dynamic candidate preparation completed: {audit}', flush=True)
    return DynamicData(dates,cash,masks,entries,catalog,features,market,paths,cycle_map,audit,start)
