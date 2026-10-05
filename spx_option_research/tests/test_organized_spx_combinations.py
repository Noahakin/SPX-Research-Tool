from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import organized_spx_combinations as model


class PremiumSizingTests(unittest.TestCase):
    def test_budget_uses_net_credit_and_all_in_hedge_cost(self):
        # Gross spread credit3.00, spread execution/fees .13; hedge price3.00
        # plus .065 execution/fee. Exactly25% of2.87 funds the hedge.
        q, ok = model.premium_quantity(2.87,3.065,.25)
        self.assertTrue(ok)
        self.assertAlmostEqual(float(q)*3.065,.7175)
        self.assertAlmostEqual(2.87-float(q)*3.065,2.1525)

    def test_unfunded_or_nonpositive_debit_trade_is_rejected(self):
        q,ok=model.premium_quantity([0,-1,1,1,np.nan],[1,1,0,-1,1],.25)
        np.testing.assert_array_equal(ok,np.zeros(5,bool))
        np.testing.assert_array_equal(q,np.zeros(5))

    def test_hedge_quantity_varies_with_price_not_notional(self):
        q,ok=model.premium_quantity([2,4,4],[1,1,2],.25)
        np.testing.assert_allclose(q,[.5,1,.5])
        self.assertTrue(ok.all())

    def test_penny_hedge_spend_stops_at_notional_cap(self):
        q,ok=model.premium_quantity(100,.005,.50)
        self.assertTrue(ok)
        self.assertEqual(float(q),1)
        self.assertLess(float(q)*.005,50)
        q2,_=model.premium_quantity(100,.005,.50,cap=2)
        self.assertEqual(float(q2),2)


class CombinationAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.output=Path(cls.temp.name)
        cls.days=pd.bdate_range("2024-01-05","2024-01-23")
        cls.cash=pd.Series([1000,1000,920,980,995,1005,1008,1010,990,1005,1020,1010,1030],index=cls.days,dtype=float)
        rows,entries,quotes=[],[],[]
        for n,(a,b,traded) in enumerate((("2024-01-05","2024-01-12",True),("2024-01-12","2024-01-19",False),("2024-01-19","2024-01-26",True))):
            a,b=pd.Timestamp(a),pd.Timestamp(b)
            roll=f"r{n}"
            rows.append(dict(roll_id=roll,entry_date=a,expiration_date=b,valuation_end=min(b,cls.days[-1]),
                             target_dte=7,actual_dte=7,has_expiry=traded))
            if not traded:
                continue
            spot=cls.cash.loc[a]
            for ratio in range(80,111):
                k=ratio/100*spot
                symbol=f"{roll}put{ratio}"
                entries.append(dict(roll_id=roll,ratio=ratio/100,strike=k,option_symbol=symbol))
                for day in cls.days[(cls.days>=a)&(cls.days<b)]:
                    extrinsic=max(.10,8-abs(ratio-100))*(b-day).days/7
                    mid=max(k-cls.cash.loc[day],0)+extrinsic
                    # Keep the narrow penny-option ask/bid symmetric around
                    # the midpoint while ensuring nonnegative observed bids.
                    half=min(.10,mid/2)
                    quotes.append(dict(snapshot_date=day,option_symbol=symbol,used_mid=mid,
                                       used_bid=mid-half,used_ask=mid+half,estimated=False))
        pd.DataFrame(rows).to_parquet(cls.output/"schedule.parquet")
        pd.DataFrame(entries).to_parquet(cls.output/"entries.parquet")
        model.simulate(cls.output,quotes=pd.DataFrame(quotes),cash=cls.cash,write_combined=False)
        cls.ledger=pd.read_parquet(cls.output/"trade_ledger.parquet")
        cls.metrics=pd.read_csv(cls.output/"combination_metrics.csv").set_index("strategy_id")
        z=np.load(cls.output/"combination_curves.npz")
        cls.curves=pd.DataFrame(z["equity"],index=cls.days,columns=z["strategy_ids"])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_shared_middle_leg_cost_is_scaled_by_hedge_contracts(self):
        key="combo_98-95_h95_p25_d07"
        q=.25*2.87/3.065
        expected=1e6*(1-(.13+q*.065)/1000)
        self.assertAlmostEqual(self.curves[key].iloc[0],expected)
        row=self.ledger[self.ledger.strategy_id.eq(key)].iloc[0]
        self.assertAlmostEqual(row.hedge_quantity_ratio,q)
        self.assertAlmostEqual(row.net_credit_retained_points,2.1525)

    def test_expiring_worthless_leaves_retained_net_premium(self):
        key="combo_98-95_h95_p25_d07"
        self.assertAlmostEqual(self.curves.loc["2024-01-12",key],1e6*(1+2.1525/1000))

    def test_downside_put_pnl_uses_the_fixed_entry_quantity(self):
        key="combo_98-95_h95_p25_d07"
        q=.25*2.87/3.065
        # OnJan9 spot920:98put=60+6*3/7;95put=30+3*3/7.
        high,low=60+6*3/7,30+3*3/7
        expected=1e6*(1+(2.87-(high-low)+q*(low-3.065))/1000)
        self.assertAlmostEqual(self.curves.loc["2024-01-09",key],expected)

    def test_option_cash_cycle_has_no_underlying_exposure(self):
        key="combo_98-95_h95_p25_d07"
        self.assertAlmostEqual(self.curves.loc["2024-01-17",key],self.curves.loc["2024-01-12",key])
        self.assertEqual(self.metrics.loc[key,"underlying"],0)

    def test_roll_uses_settled_equity_before_next_entry_cost(self):
        key="combo_98-95_h95_p25_d07"
        settled=1e6*(1+2.1525/1000)
        q=.25*2.87/3.065
        self.assertAlmostEqual(self.curves.loc["2024-01-19",key],settled*(1-(.13+q*.065)/1020))

    def test_buffer_has_three_net_contract_strikes_and_budget_reconciles(self):
        key="combo_98-95_h95-85_p50_d07"
        row=self.ledger[self.ledger.strategy_id.eq(key)].iloc[0]
        #85put quoted .05/.15: sale .075 less .015 commission = .06.
        expected_debit=3.065-.06
        self.assertAlmostEqual(row.hedge_debit_points,expected_debit)
        self.assertAlmostEqual(row.hedge_quantity_ratio*expected_debit,.5*2.87)
        self.assertEqual(row.hedge_secondary_strike,850)

    def test_final_unexpired_positions_use_option_marks(self):
        key="combo_98-95_h95_p25_d07"
        last=self.ledger[self.ledger.strategy_id.eq(key)].iloc[-1]
        q=.25*2.87/3.065
        expected=last.entry_equity*(1+(2.87-(6-3)*3/7+q*(3*3/7-3.065))/1020)
        self.assertTrue(last.final_position_open)
        self.assertAlmostEqual(self.curves[key].iloc[-1],expected)


if __name__=="__main__":
    unittest.main()
