from maturity_grid_engine import trade_paths
import numpy as np

def inspect(prepared):
    records=[]
    dates=prepared[0]
    for c in prepared[3][60]:
        if not c.cadence[3]: continue
        t=trade_paths(c,[103.],np.array([3.]),np.array([.25]))
        if not t['entry_valid'][0]: continue
        hi,lo=46,40
        limit=int(t['offsets'][0,0])
        raw=c.mid[:,hi]-c.mid[:,lo]; width=c.strikes[hi]-c.strikes[lo]
        bad=(~np.isfinite(raw))|(raw<-.011)|(raw>width+.011)|(c.quality[:,hi]==0)|(c.quality[:,lo]==0)
        for k in np.flatnonzero(bad[:limit+1]):
            records.append(dict(entry=str(dates[c.entry].date()),date=str(dates[c.entry+k].date()),raw=float(raw[k]),width=float(width),short_mid=float(c.mid[k,hi]),long_mid=float(c.mid[k,lo]),quality=(int(c.quality[k,hi]),int(c.quality[k,lo]))))
    return records
