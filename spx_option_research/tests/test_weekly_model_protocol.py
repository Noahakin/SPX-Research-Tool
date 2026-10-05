from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_weekly_dynamic_research.py"
SPEC = importlib.util.spec_from_file_location("weekly_dynamic_research", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WeeklyProtocolTests(unittest.TestCase):
    def test_primary_choices_exclude_iv_z_diagnostic_and_always_trade_controls(self):
        rows = []
        for universe in ("width3", "adaptive_width"):
            for model, gate, sharpe in (("ridge__core", "positive_edge", 1.0),
                                        ("rule__two_leg_iv_z", "positive_edge", 100.0),
                                        ("ridge__full", "always", 50.0)):
                rows.append(dict(model=model, universe=universe, gate=gate, sharpe=sharpe,
                                 cagr=0.02, active_weeks=100, policy_id=f"{model}_{universe}_{gate}"))
        rows.append(dict(model="fixed__0.99", universe="width3", gate="always", sharpe=0.5,
                         cagr=0.01, active_weeks=100, policy_id="fixed"))
        table = pd.DataFrame(rows)
        info = table.set_index("policy_id")[["model", "universe", "gate"]].to_dict("index")
        with tempfile.TemporaryDirectory() as tmp, patch.object(MODULE, "OUT", Path(tmp)):
            table.to_csv(Path(tmp) / "validation_policy_results.csv", index=False)
            (Path(tmp) / "research_protocol.json").write_text("{}", encoding="utf-8")
            frozen = MODULE.freeze_choices(table, info)
        for universe in ("width3", "adaptive_width"):
            self.assertEqual(frozen["choices"][f"Dynamic winner: {universe}"]["model"], "ridge__core")

    def test_validation_excludes_trade_whose_payoff_occurs_in_holdout(self):
        dates = pd.to_datetime(["2023-12-22", "2023-12-29"])
        frame = pd.DataFrame({"candidate_id": ["a", "b"], "entry_date": dates,
            "expiration_date": dates + pd.Timedelta(days=7), "target_width_pct": [0.03, 0.03],
            "target_short_ratio": [0.99, 0.99]})
        pred = frame[["candidate_id"]].assign(model_score=1.0)
        selected = MODULE.choose_policy(frame, pred, "model_score", "width3", "always",
                                         start=pd.Timestamp("2023-01-01"), end=pd.Timestamp("2024-01-01"))
        self.assertEqual(selected.candidate_id.tolist(), ["a"])

    def test_hedge_width_choice_and_no_trade_gate_use_scores_not_outcomes(self):
        entry = pd.Timestamp("2022-01-07")
        frame = pd.DataFrame({"candidate_id": ["a", "b", "c"], "entry_date": [entry] * 3,
            "expiration_date": [entry + pd.Timedelta(days=7)] * 3,
            "target_width_pct": [0.03, 0.03, 0.01], "target_short_ratio": [0.98, 1.01, 0.99],
            "pnl_realistic_pct_spot_notional": [10.0, -10.0, -100.0]})
        pred = pd.DataFrame({"candidate_id": ["a", "b", "c"], "model_score": [0.1, 0.2, 0.5]})
        fixed = MODULE.choose_policy(frame, pred, "model_score", "width3", "positive_edge", start=entry)
        adaptive = MODULE.choose_policy(frame, pred, "model_score", "adaptive_width", "positive_edge", start=entry)
        self.assertEqual(fixed.candidate_id.tolist(), ["b"])
        self.assertEqual(adaptive.candidate_id.tolist(), ["c"])
        pred.model_score = -1.0
        cash = MODULE.choose_policy(frame, pred, "model_score", "width3", "positive_edge", start=entry)
        self.assertEqual(len(cash), 1)
        self.assertTrue(cash.candidate_id.isna().all())

    def test_yearly_refit_uses_only_matured_prior_year_labels(self):
        dates = pd.date_range("2017-01-06", "2021-12-31", freq="W-FRI")
        frame = pd.DataFrame({"candidate_id": [f"candidate_{i}" for i in range(len(dates))],
            "entry_date": dates, "expiration_date": dates + pd.Timedelta(days=7),
            "target_liability_fraction": np.arange(len(dates), dtype=float), "x": np.arange(len(dates), dtype=float),
            "premium_realistic_pct_spot_notional": 0.001, "width_pct": 0.03,
            "econ_risk_denominator": 0.005})
        future = frame.expiration_date.ge(pd.Timestamp("2021-01-01"))
        frame.loc[future, "target_liability_fraction"] = 999999.0
        calls = []

        class Model:
            def predict(self, x):
                return np.full(len(x), 0.1)

        def mocked_fit(family, x, y, weights):
            calls.append((x.copy(), y.copy(), weights.copy()))
            return Model()

        with tempfile.TemporaryDirectory() as tmp, patch.object(MODULE, "OUT", Path(tmp)), patch.object(MODULE, "fit_model", mocked_fit):
            pred = MODULE.predict_years(frame, {"core": ["x"]}, [2021], save_prefix="test")
            self.assertTrue(calls)
            for x, y, weights in calls:
                self.assertLess(y.max(), 999999.0)
                self.assertTrue(frame.loc[x.index, "expiration_date"].lt(pd.Timestamp("2021-01-01")).all())
                self.assertTrue(weights.eq(1.0).all())
            values = pred.loc[pred.entry_date.dt.year.eq(2021), "ridge__core"]
            np.testing.assert_allclose(values, (0.001 - 0.1 * 0.03) / 0.005)


if __name__ == "__main__":
    unittest.main()
