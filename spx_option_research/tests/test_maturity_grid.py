import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from maturity_grid_data import Cycle,RATIOS,entry_calendar,nearest_indices
from maturity_grid_engine import trade_paths,simulate,INITIAL


def example_cycle(entry=0,short_values=(2,1.4,1),short_cost=.1,long_cost=.1,valid=True):
    n=len(short_values);mid=np.zeros((n,len(RATIOS)));cost=np.full_like(mid,long_cost)
    hi=int((103-80)*2);lo=int((100-80)*2)
    mid[:,hi]=short_values;cost[:,hi]=short_cost
    quality=np.ones_like(mid,dtype=np.uint8)
    if not valid:mid[0,hi]=np.nan;cost[0,hi]=np.nan;quality[0,hi]=0
    return Cycle(3,entry,entry+n-1,pd.Timestamp('2024-01-31'),3,100.,np.array([1,1,1,1],bool),
        RATIOS.copy(),np.array([str(v) for v in RATIOS]),np.zeros(len(RATIOS)),mid,cost,quality,False)


class GridTests(unittest.TestCase):
    def test_nearest_strike_ties_choose_lower(self):
        np.testing.assert_array_equal(nearest_indices(np.array([90,100,110]),np.array([95,109,120])),[0,2,2])

    def test_profit_exit_uses_roundtrip_costs_and_no_same_day_profit(self):
        c=example_cycle()
        t=trade_paths(c,[103],np.array([3]),np.array([.25]))
        # Net credit 180. Day 1 liquidation profit is 20, below the 45 target.
        # Day 2 earns 60 after closing costs and qualifies.
        self.assertEqual(t['offsets'][0,0],2)
        self.assertAlmostEqual(t['terminal'][0,0],60)
        np.testing.assert_allclose(t['increments'][0,0],[-20,60,20])
        self.assertAlmostEqual(t['increments'][0,0].sum(),60)

    def test_valuations_estimated_from_quotes_cannot_trigger_profit_taking(self):
        c=example_cycle(short_values=(2,.2,.9))
        c.quality[1,int((103-80)*2)]=2
        t=trade_paths(c,[103],np.array([3]),np.array([.25]))
        self.assertEqual(t['offsets'][0,0],2)
        self.assertEqual(t['estimated'][0,0],1)

    def test_missing_entry_does_not_poison_future_equity(self):
        dates=pd.date_range('2024-01-02',periods=6,freq='B');masks=np.ones((6,4),bool)
        curves,meta,_=simulate([example_cycle(valid=False),example_cycle(entry=3)],[103],dates,masks,np.array([3]),np.array([.25]))
        self.assertTrue(np.isfinite(curves).all());self.assertTrue((meta['trades']==1).all())
        # Current-equity notional is one million / (100*100) = 100 contracts.
        self.assertAlmostEqual(curves[1,-1],INITIAL+6000)

    def test_overlapping_entries_respect_risk_and_notional_caps(self):
        dates=pd.date_range('2024-01-02',periods=5,freq='B');masks=np.ones((5,4),bool)
        curves,meta,rows=simulate([example_cycle(),example_cycle(entry=1)],[103],dates,masks,np.array([3]),np.array([.25]),True)
        self.assertTrue((meta['max_risk_pct_initial'][::2]<=.05+1e-12).all())
        self.assertTrue((meta['max_notional_pct_initial'][1::2]<=1+1e-12).all())
        for i in range(len(curves)):
            realized=sum(r['realized_pnl'] for r in rows if r['index']==i)
            self.assertAlmostEqual(curves[i,-1],INITIAL+realized)

    def test_unresolved_held_prices_are_flagged(self):
        c=example_cycle();c.mid[1,int((103-80)*2)]=np.nan;c.quality[1,int((103-80)*2)]=0
        t=trade_paths(c,[103],np.array([3]),np.array([.25]))
        self.assertGreater(t['unresolved'][0,0],0)
        self.assertTrue(np.isfinite(t['increments']).all())

    def test_wide_quote_midpoint_preserved_and_flagged_when_market_is_feasible(self):
        c=example_cycle(short_values=(2,3.1,.9))
        t=trade_paths(c,[103],np.array([3]),np.array([.25]))
        self.assertEqual(t['unresolved'][0,0],0)
        self.assertEqual(t['bound_flags'][0,0],1)
        self.assertAlmostEqual(t['increments'][0,0,:2].sum(),180-310)
        self.assertEqual(t['offsets'][0,0],2)

    def test_current_equity_sizing_compounds_at_later_entries(self):
        dates=pd.date_range('2024-01-02',periods=6,freq='B');masks=np.ones((6,4),bool)
        curves,_,_=simulate([example_cycle(),example_cycle(entry=3)],[103],dates,masks,np.array([3]),np.array([.25]))
        self.assertAlmostEqual(curves[1,-1],INITIAL*(1.006**2))


if __name__=='__main__':unittest.main()
