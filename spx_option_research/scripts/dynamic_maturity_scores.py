"""At-entry scoring rules: no realized trade outcomes are accepted as inputs."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.special import ndtr


def vertical_moments(spot, high, low, years, sigma, drift):
    """Exact first two payoff moments of a put vertical under lognormal SPX.

    This is a physical terminal-payoff model used as an entry proxy. It does
    not assert that terminal expected P&L equals the expected 25%-exit P&L.
    No discounting is applied: the study earns zero on idle cash.
    """
    variance = sigma*sigma*years
    sd = np.sqrt(variance)
    location = np.log(spot)+(drift-.5*sigma*sigma)*years
    def moment_below(strike, power):
        z = (np.log(strike)-location-power*variance)/sd
        return np.exp(power*location+.5*power*power*variance)*ndtr(z)
    p_high, p_low = moment_below(high,0), moment_below(low,0)
    s_high, s_low = moment_below(high,1), moment_below(low,1)
    s2_high, s2_low = moment_below(high,2), moment_below(low,2)
    mean = high*p_high-s_high-(low*p_low-s_low)
    second = (high-low)**2*p_low+high**2*(p_high-p_low)-2*high*(s_high-s_low)+(s2_high-s2_low)
    return mean, np.sqrt(np.maximum(second-mean*mean,0))


def prior_zscore(values, window):
    """Each candidate's own preceding weekly observations; current row excluded."""
    frame = pd.DataFrame(values)
    history = frame.shift(1).rolling(window, min_periods=int(np.ceil(.8*window)))
    mean, sd = history.mean().to_numpy(), history.std(ddof=1).to_numpy()
    return np.divide(values-mean, sd, out=np.full_like(values,np.nan), where=sd>1e-10)


def score_rules(features, market, spot):
    """Yield 88 predetermined scoring specifications with only entry features."""
    f = features
    years = f['dte']/365.2425
    short, long = f['short_strike'], f['long_strike']
    spot = np.asarray(spot)[:,None]
    for rv, multiplier, drift in itertools.product((21,63,126),(.8,1.,1.2),(0.,.05)):
        sigma = market[f'rv{rv}'].to_numpy()[:,None]*multiplier
        mean, deviation = vertical_moments(spot,short,long,years,sigma,drift)
        edge = f['credit']-100*mean-f['closing_cost']
        deviation = np.maximum(100*deviation, f['closing_cost'])
        choices = {
            'edge_per_risk': edge/f['risk'],
            'annual_edge_per_risk': edge/f['risk']/years,
            'payoff_sharpe': edge/deviation,
            'annual_payoff_sharpe': edge/deviation/np.sqrt(years),
        }
        for method, score in choices.items():
            yield dict(rule=f'{method}_rv{rv}_scale{multiplier:g}_drift{drift:g}',
                family=method, rv_window=rv, vol_multiplier=multiplier, assumed_drift=drift,
                history_window=0, score=score)
    for method, score in {
        'credit_per_risk': f['credit']/f['risk'],
        'annual_credit_per_risk': f['credit']/f['risk']/years,
        'credit_per_width': f['credit']/((short-long)*100),
        'annual_credit_per_notional': f['credit']/(spot*100)/years,
    }.items():
        yield dict(rule=method,family=method,rv_window=0,vol_multiplier=1.,assumed_drift=0.,history_window=0,score=score)
    for rv, window, iv_kind in itertools.product((21,63),(52,104),('short','mean_leg')):
        iv = f['short_iv'] if iv_kind == 'short' else (f['short_iv']+f['long_iv'])/2
        ratio = iv/market[f'rv{rv}'].to_numpy()[:,None]
        ratio = np.where(f['observed_entry'],ratio,np.nan)
        yield dict(rule=f'{iv_kind}_iv_rv_z_rv{rv}_history{window}',family='iv_rv_history_z',
            rv_window=rv,vol_multiplier=1.,assumed_drift=0.,history_window=window,
            score=prior_zscore(ratio,window))
    for rv, window in itertools.product((21,63),(52,104)):
        sigma = market[f'rv{rv}'].to_numpy()[:,None]
        mean, _ = vertical_moments(spot,short,long,years,sigma,0.)
        residual = (f['credit']-100*mean-f['closing_cost'])/((short-long)*100)
        residual = np.where(f['observed_entry'],residual,np.nan)
        yield dict(rule=f'whole_spread_edge_z_rv{rv}_history{window}',family='whole_spread_history_z',
            rv_window=rv,vol_multiplier=1.,assumed_drift=0.,history_window=window,
            score=prior_zscore(residual,window))


def universes(catalog):
    rows, masks = [], []
    maturities = [(3,14),(14,45),(28,90),(45,120),(90,180),(3,180)]
    strikes = [(90,100),(98,105),(90,110)]
    widths = [(1,3),(3,5),(5,10),(1,10)]
    for maturity, strike, width in itertools.product(maturities,strikes,widths):
        row = dict(universe=f'dte{maturity[0]}-{maturity[1]}_short{strike[0]}-{strike[1]}_width{width[0]}-{width[1]}',
            min_dte=maturity[0],max_dte=maturity[1],min_short=strike[0],max_short=strike[1],min_width=width[0],max_width=width[1])
        masks.append((catalog.target_dte.between(*maturity)&catalog.short_target.between(*strike)&catalog.width.between(*width)).to_numpy())
        rows.append(row)
    return rows, masks


def choose(score, valid, universe_masks):
    """Return the highest signed score, with and without a positive-score gate.

    Stable candidate order breaks ties by target DTE, short strike, then width.
    The function cannot access future marks, exits, or realized returns.
    """
    selections, values, counts = [], [], []
    eligible = valid & np.isfinite(score)
    for mask in universe_masks:
        candidates = np.flatnonzero(mask)
        values_in = np.where(eligible[:,candidates],score[:,candidates],-np.inf)
        local = values_in.argmax(axis=1)
        best_score = values_in[np.arange(len(score)),local]
        best = candidates[local]
        count = eligible[:,candidates].sum(axis=1)
        for positive in (False,True):
            ok = np.isfinite(best_score)&((best_score>0) if positive else True)
            selections.append(np.where(ok,best,-1))
            values.append(best_score)
            counts.append(count)
    return np.asarray(selections,dtype=np.int32), np.asarray(values), np.asarray(counts,dtype=np.int16)
