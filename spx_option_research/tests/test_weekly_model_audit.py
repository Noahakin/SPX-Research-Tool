"""Independent chronology and row-alignment checks for the weekly model driver."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_weekly_dynamic_research.py"
SPEC = importlib.util.spec_from_file_location("weekly_dynamic_model_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def model_frame():
    prior_entries = pd.date_range(end="2020-12-25", periods=160, freq="W-FRI")
    dates = prior_entries.append(pd.DatetimeIndex(["2021-01-01", "2021-01-08"]))
    rows = []
    for i, date in enumerate(dates):
        for j in range(2):
            width = (0.01, 0.05)[j]
            rows.append({
                "candidate_id": f"week{i}_pair{j}", "entry_date": date,
                "expiration_date": date + pd.Timedelta(days=7),
                "feature": (0.25, 0.75)[j],
                "target_liability_fraction": 0.1 + i / 1000,
                "width_pct": width, "premium_realistic_pct_spot_notional": width * 0.5,
                "econ_risk_denominator": (0.002, 0.005)[j],
            })
    return pd.DataFrame(rows).sample(frac=1, random_state=123)


class Predictor:
    def predict(self, features):
        return features.feature.to_numpy()


class WeeklyModelAuditTests(unittest.TestCase):
    def test_annual_cutoff_week_weighting_and_shuffled_prediction_alignment(self):
        frame = model_frame()
        calls = []

        def fake_fit(family, x, y, weights):
            calls.append((family, x.copy(), y.copy(), weights.copy()))
            return Predictor()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(MODULE, "fit_model", side_effect=fake_fit), patch.object(MODULE, "OUT", Path(directory)):
                predicted = MODULE.predict_years(frame, {"core": ["feature"]}, [2021], save_prefix="audit")
            audit = pd.read_csv(Path(directory) / "audit_fit_audit.csv", parse_dates=["latest_training_expiration"])
        cutoff = pd.Timestamp("2021-01-01")
        self.assertTrue(audit.latest_training_expiration.lt(cutoff).all())
        self.assertTrue(audit.training_weeks.eq(159).all())
        self.assertEqual(len(calls), 4)
        for _, x, y, weights in calls:
            self.assertEqual(x.columns.tolist(), ["feature"])
            self.assertTrue(frame.loc[x.index, "expiration_date"].lt(cutoff).all())
            pd.testing.assert_series_equal(y, frame.loc[x.index, "target_liability_fraction"])
            np.testing.assert_allclose(weights.groupby(frame.loc[x.index, "entry_date"]).sum(), 1.0)
        test = frame.loc[frame.entry_date.ge(cutoff)].set_index("candidate_id")
        score = ((test.premium_realistic_pct_spot_notional - test.feature * test.width_pct)
                 / test.econ_risk_denominator)
        values = predicted.set_index("candidate_id")
        for column in ("ridge__core", "ridge__full", "boosting__core", "boosting__full"):
            np.testing.assert_allclose(values.loc[score.index, column], score)
        self.assertTrue(values.loc[frame.loc[frame.entry_date.lt(cutoff), "candidate_id"], "ridge__core"].isna().all())

    def test_current_and_future_outcomes_cannot_change_fitted_predictions(self):
        frame = model_frame()
        changed = frame.copy()
        changed.loc[changed.expiration_date.ge("2021-01-01"), "target_liability_fraction"] = 0.999
        training_targets = []

        def fake_fit(family, x, y, weights):
            training_targets.append(y.copy())
            return Predictor()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(MODULE, "fit_model", side_effect=fake_fit), patch.object(MODULE, "OUT", Path(directory)):
                before = MODULE.predict_years(frame, {"core": ["feature"]}, [2021], save_prefix="before")
                after = MODULE.predict_years(changed, {"core": ["feature"]}, [2021], save_prefix="after")
        pd.testing.assert_frame_equal(before, after)
        for left, right in zip(training_targets[:4], training_targets[4:]):
            pd.testing.assert_series_equal(left, right)

    def test_positive_edge_gate_and_inclusive_risk_threshold(self):
        rows = []
        scores = []
        for week, date in enumerate(pd.date_range("2021-01-01", periods=3, freq="W-FRI")):
            for leg, short in enumerate((0.98, 0.99)):
                candidate_id = f"week{week}_{leg}"
                rows.append({"candidate_id": candidate_id, "entry_date": date,
                             "expiration_date": date + pd.Timedelta(days=7),
                             "target_width_pct": 0.03, "target_short_ratio": short})
                scores.append({"candidate_id": candidate_id,
                               "ridge__core": ((0.10, 0.05), (-0.1, -0.2), (0.0, -0.1))[week][leg]})
        frame, predictions = pd.DataFrame(rows), pd.DataFrame(scores)
        positive = MODULE.choose_policy(frame, predictions, "ridge__core", "width3", "positive_edge", start=pd.Timestamp("2021-01-01"))
        stricter = MODULE.choose_policy(frame, predictions, "ridge__core", "width3", "edge_risk_0.10", start=pd.Timestamp("2021-01-01"))
        always = MODULE.choose_policy(frame, predictions, "ridge__core", "width3", "always", start=pd.Timestamp("2021-01-01"))
        self.assertEqual(positive.candidate_id.iloc[0], "week0_0")
        self.assertTrue(positive.candidate_id.iloc[1:].isna().all())
        self.assertEqual(stricter.candidate_id.iloc[0], "week0_0")
        self.assertTrue(stricter.candidate_id.iloc[1:].isna().all())
        self.assertEqual(always.candidate_id.tolist(), ["week0_0", "week1_0", "week2_0"])


if __name__ == "__main__":
    unittest.main()
