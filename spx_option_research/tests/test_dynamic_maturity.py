from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from dynamic_maturity_scores import vertical_moments, prior_zscore, choose
from dynamic_maturity_data import market_features
from dynamic_maturity_engine import simulate, profiles, chosen_increments
from maturity_grid_data import Cycle, RATIOS, SHORTS, WIDTHS, TENORS
from maturity_grid_engine import trade_paths, simulate as fixed_simulate


class DynamicMaturityTests(unittest.TestCase):
    def test_payoff_moments_match_numerical_integration(self):
        spot, high, low, years, sigma, drift = 100.,103.,98.,42/365.2425,.24,.05
        mean, sd = vertical_moments(spot,high,low,years,sigma,drift)
        def payoff(z):
            underlying=spot*np.exp((drift-.5*sigma*sigma)*years+sigma*np.sqrt(years)*z)
            return max(high-underlying,0)-max(low-underlying,0)
        breaks=[-12,*[(np.log(k/spot)-(drift-.5*sigma*sigma)*years)/(sigma*np.sqrt(years)) for k in (low,high)],12]
        first=sum(quad(lambda z:payoff(z)*norm.pdf(z),a,b,epsabs=1e-10)[0] for a,b in zip(breaks[:-1],breaks[1:]))
        second=sum(quad(lambda z:payoff(z)**2*norm.pdf(z),a,b,epsabs=1e-10)[0] for a,b in zip(breaks[:-1],breaks[1:]))
        self.assertAlmostEqual(float(mean),first,places=9)
        self.assertAlmostEqual(float(sd),np.sqrt(second-first*first),places=8)

    def test_joint_choice_can_switch_geometry_and_expiry(self):
        score=np.array([[1.,3.,2.],[4.,1.,2.],[1.,2.,5.]])
        selected,_,_=choose(score,np.ones_like(score,dtype=bool),[np.ones(3,dtype=bool)])
        self.assertEqual(selected[0].tolist(),[1,0,2])

    def test_positive_gate_and_unavailable_candidates(self):
        score=np.array([[-1.,-2.],[10.,1.]])
        valid=np.array([[True,True],[False,True]])
        selected,_,counts=choose(score,valid,[np.ones(2,dtype=bool)])
        self.assertEqual(selected.tolist(),[[0,1],[-1,1]])
        self.assertEqual(counts[0].tolist(),[2,1])

    def test_history_excludes_current_and_future_observations(self):
        values=np.arange(1.,161.)[:,None]
        baseline=prior_zscore(values,52)
        changed=values.copy();changed[120:]*=100
        np.testing.assert_allclose(baseline[:120],prior_zscore(changed,52)[:120],equal_nan=True)
        expected=(values[100,0]-values[48:100,0].mean())/values[48:100,0].std(ddof=1)
        self.assertAlmostEqual(baseline[100,0],expected)

    def test_cash_features_exclude_entry_close(self):
        dates=pd.bdate_range('2020-01-01',periods=300)
        cash=pd.Series(100*np.exp(np.sin(np.arange(300)/10)*.1+np.arange(300)*.001),index=dates)
        baseline=market_features(cash,dates)
        changed=cash.copy();changed.iloc[260:]*=2
        pd.testing.assert_frame_equal(baseline.iloc[:261],market_features(changed,dates).iloc[:261])

    def test_dynamic_fixed_choice_reconciles_to_existing_engine(self):
        dates=pd.bdate_range('2020-01-03',periods=21)
        masks=np.zeros((len(dates),4),bool);masks[[0,5,10],:]=True
        mid=np.maximum(RATIOS-100,0)[None,:]*np.linspace(1,.4,11)[:,None]+.5
        cycles=[]
        for entry in (0,5,10):
            cycles.append(Cycle(14,entry,entry+10,dates[entry+10],14,100.,masks[entry],RATIOS.copy(),
                np.array([str(x) for x in RATIOS]),np.zeros(len(RATIOS)),mid.copy(),
                np.full_like(mid,.015),np.ones_like(mid,dtype=np.uint8),False))
        ti=TENORS.index(14);g=list(SHORTS).index(103.)*len(WIDTHS)+list(WIDTHS).index(3.)
        candidate=ti*len(SHORTS)*len(WIDTHS)+g
        catalog=pd.DataFrame([dict(target_dte=t,short_target=s,width=w) for t in TENORS for s in SHORTS for w in WIDTHS])
        paths={}
        for row,cycle in enumerate(cycles):
            path=trade_paths(cycle,SHORTS,WIDTHS,np.array([.25]));path['selection_unresolved']=np.zeros(len(SHORTS)*len(WIDTHS),int)
            original_increments=path.pop('increments')
            np.testing.assert_allclose(chosen_increments(cycle,path,np.array([g])),original_increments[[g],0,:],rtol=0,atol=1e-10)
            paths[(row,ti)]=path
        data=SimpleNamespace(dates=dates,start=0,entries=np.array([0,5,10]),masks=masks,paths=paths,
            cycles={(i,ti):c for i,c in enumerate(cycles)},catalog=catalog,
            features={'valid':np.ones((3,len(catalog)),bool),'closing_cost':np.full((3,len(catalog)),3.)})
        selected=np.full((1,3),candidate)
        actual,meta,_=simulate(data,selected)
        expected,_,_=fixed_simulate(cycles,[103.],dates,masks,np.array([3.]),np.array([.25]))
        profile=profiles(1)
        for i,r in profile.loc[profile.mode_index.eq(0)].iterrows():
            np.testing.assert_allclose(actual[i],expected[r.cadence_index*2+r.sizing_index],rtol=0,atol=1e-7)
        self.assertTrue((meta['unique_maturity_targets']==1).all())


if __name__=='__main__':
    unittest.main()
