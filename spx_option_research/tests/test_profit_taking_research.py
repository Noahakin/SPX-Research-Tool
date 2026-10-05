import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from profit_taking_research import simulate, choose_expiry, Snapshot, QuoteStore, entry_packages, missing_ask_estimate, INITIAL


def record(category='Put selling',target=.25,**updates):
    row=dict(id='test_tp25',baseId='test',profitTarget=target,category=category,days=3,
             primary=100,width=0,premium=None,dynamic=False,hedgeHigh=None)
    row.update(updates)
    return row


def snapshots(dates,time_values,expiry='2024-01-12'):
    expiry=pd.Timestamp(expiry)
    for day,value in zip(dates,time_values):
        rows=[]
        for strike in range(800,1101,10):
            mid=max(strike-1000.,0)+value
            rows.append(dict(snapshot_date=day,expiration_date=expiry,option_type='put',
                option_symbol=f'SPXW_{expiry:%Y%m%d}_{strike}',strike=float(strike),bid=mid,ask=mid))
        yield pd.DataFrame(rows)


class ProfitTakingTests(unittest.TestCase):
    def test_expiry_ties_prefer_longer_and_no_past_or_weekend_expiry(self):
        dates=pd.date_range('2024-01-02',periods=10,freq='B')
        expiries=pd.to_datetime(['2024-01-02','2024-01-04','2024-01-06','2024-01-08'])
        self.assertEqual(choose_expiry(expiries,dates[0],3,dates),pd.Timestamp('2024-01-04'))
        self.assertIsNone(choose_expiry(expiries,dates[0],56,dates))

    def run_case(self,values,category='Put selling',target=.25,known=None,expiry='2024-01-05',cash=None):
        dates=pd.date_range('2024-01-02',periods=len(values),freq='B')
        if cash is None:cash=np.full(len(dates),1000.)
        result=simulate([record(category,target)],dates,np.asarray(cash),snapshots(dates,values,expiry),known)
        return dates,result

    def test_whole_profit_exit_reenters_same_session_and_charges_both_costs(self):
        _,(nav,histories,_,meta,_)=self.run_case([20.,10.,10.])
        trades=np.array(histories[0]).reshape(-1,7)
        self.assertEqual(trades[0,1],1)
        self.assertEqual(trades[0,-1],1)
        self.assertEqual(trades[1,0],1)
        # First round-trip P&L: 19.985 received minus 10.015 to close.
        settled=INITIAL*(1+9.97/1000)
        self.assertAlmostEqual(nav[0,0,1],settled*(1-.015/1000))
        self.assertEqual(meta['trades'][0],2)
        self.assertEqual(meta['profitExits'][0],1)

    def test_profit_trigger_includes_closing_costs(self):
        _,(_,histories,_,meta,_)=self.run_case([20.,10.,10.],target=.5)
        # 9.97 net profit is below 50% of 19.985 net credit.
        self.assertEqual(meta['profitExits'][0],0)
        self.assertEqual(len(histories[0]),7)

    def test_long_put_uses_profit_relative_to_debit(self):
        _,(nav,histories,_,meta,_)=self.run_case([20.,30.,30.],category='Put buying')
        settled=INITIAL*(1+9.97/1000)
        self.assertAlmostEqual(nav[0,0,1],settled*(1-.015/1000))
        self.assertEqual(meta['profitExits'][0],1)
        self.assertEqual(np.array(histories[0]).reshape(-1,7)[1,0],1)

    def test_estimated_mark_values_position_but_cannot_trigger_exit(self):
        known={(pd.Timestamp('2024-01-03'),'SPXW_20240105_1000'):(10.,.015)}
        _,(nav,histories,_,meta,_)=self.run_case([20.,10.,10.],known=known)
        trades=np.array(histories[0]).reshape(-1,7)
        self.assertEqual(trades[0,1],2)
        self.assertEqual(meta['estimatedDays'][0],1)
        self.assertAlmostEqual(nav[0,0,1],INITIAL*(1+9.985/1000))

    def test_expiry_settles_intrinsic_without_closing_fees(self):
        _,(nav,histories,_,meta,_)=self.run_case([20.,21.,22.,0.],target=.75)
        self.assertEqual(meta['expiryExits'][0],1)
        self.assertEqual(meta['profitExits'][0],0)
        self.assertAlmostEqual(nav[0,0,-1],INITIAL*(1+19.985/1000))
        self.assertEqual(histories[0][-1],2)

    def test_final_unexpired_position_is_marked_without_hypothetical_close(self):
        _,(nav,histories,_,meta,_)=self.run_case([20.,19.],target=.75)
        self.assertEqual(histories[0][-1],0)
        self.assertAlmostEqual(nav[0,0,-1],INITIAL*(1+.985/1000))
        self.assertEqual(meta['trades'][0],1)

    def test_spx_portfolio_has_own_compounding_and_continues_when_options_flat(self):
        _,(nav,_,_,meta,_)=self.run_case([20.,20.,20.],expiry='2024-02-01',cash=[1000.,1010.,1020.])
        self.assertEqual(meta['trades'][0],0)
        np.testing.assert_allclose(nav[0,0],INITIAL)
        np.testing.assert_allclose(nav[1,0],[INITIAL,INITIAL*1.01,INITIAL*1.02])

    def test_unresolved_held_quotes_fail_instead_of_forward_filling(self):
        dates=pd.date_range('2024-01-02',periods=3,freq='B')
        frames=list(snapshots(dates,[20.,20.,20.],'2024-01-05'))
        frames[1]=frames[1].iloc[:0]
        with self.assertRaisesRegex(ValueError,'Unresolved held'):
            simulate([record()],dates,np.full(3,1000.),frames)

    def test_buffer_uses_full_credit_allocation_and_preferred_five_point_width(self):
        dates=pd.date_range('2024-01-02',periods=3,freq='B')
        frame=next(snapshots(dates,[20.],'2024-01-05'))
        snap=Snapshot(frame,dates[0],[],{});store=QuoteStore()
        r=record('Both',primary=103,width=3,premium=.2,dynamic=True)
        package=entry_packages(snap,store,1000.,dates,[r])
        self.assertTrue(package['eligible'][0])
        ids=package['contracts'][0];qty=package['quantities'][0]
        self.assertEqual(store.strikes[ids[2]],1000.)
        self.assertEqual(store.strikes[ids[3]],950.)
        self.assertGreaterEqual(qty[2],1.)
        base_credit=30.-.03
        self.assertAlmostEqual(qty[2]*.03,base_credit*.2)
        self.assertAlmostEqual(package['credit'][0],base_credit*.8)

    def test_wide_donor_cannot_spoil_same_day_parity_valuation(self):
        day=pd.Timestamp('2024-01-02');expiry=pd.Timestamp('2024-02-02');rows=[]
        for strike in range(800,1201,10):
            for kind in ['put','call']:
                mid=max(strike-1000,0)+20 if kind=='put' else max(1000-strike,0)+20
                bid,ask=mid-.5,mid+.5
                if kind=='put' and strike>=1110:bid=0.
                if kind=='put' and strike==1090:bid,ask=50.,290.
                rows.append(dict(snapshot_date=day,expiration_date=expiry,option_type=kind,
                    option_symbol=f'SPXW_{kind}_{strike}',strike=float(strike),bid=bid,ask=ask))
        audit=[];snap=Snapshot(pd.DataFrame(rows),day,audit,{})
        value=snap.quote('SPXW_put_1150',1150.,expiry)
        self.assertAlmostEqual(value[0],170.,places=7)
        self.assertEqual(value[2],2)
        self.assertIn('broad donor markets excluded',audit[-1]['reason'])

    def test_missing_ask_retains_bid_and_estimates_spread_from_same_expiry_bracket(self):
        chain=pd.DataFrame(dict(option_type=['put']*3,option_symbol=['a','b','c'],strike=[5200.,5300.,5500.],bid=[24.,33.5,62.4],ask=[26.,0.,64.8]))
        value=missing_ask_estimate(chain,5300.,33.5)
        self.assertAlmostEqual(value['estimated_ask'],33.5+2.+.4/3)
        self.assertAlmostEqual(value['mid'],33.5+(2.+.4/3)/2)
        self.assertIsNone(missing_ask_estimate(chain,5300.,0.))
        self.assertIsNone(missing_ask_estimate(chain,5300.,100.))


if __name__=='__main__':unittest.main()
