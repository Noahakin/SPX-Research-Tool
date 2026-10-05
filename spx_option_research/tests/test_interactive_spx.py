from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from build_interactive_spx import overlay_spx


class InteractiveOverlayTests(unittest.TestCase):
    def test_shared_roll_settles_before_next_entry_costs(self):
        # First option trade: -1% entry friction, +10% settlement.
        # Second: -2% entry friction, +20% settlement. Shared NAV is 107.8.
        nav = np.array([[99.], [107.8], [132.]])
        cycles = [dict(ix=np.array([0,1])), dict(ix=np.array([1,2]))]
        entries = [(np.array([100.]), np.array([.10])), (np.array([110.]), np.array([.20]))]
        actual = overlay_spx(nav, cycles, entries, np.array([100.,90.,99.]), initial=100.)
        np.testing.assert_allclose(actual[:,0], [99.,98.,130.])

    def test_flat_spx_reproduces_options(self):
        nav = np.array([[99.], [107.8], [132.]])
        cycles = [dict(ix=np.array([0,1])), dict(ix=np.array([1,2]))]
        entries = [(np.array([100.]),np.array([.1])), (np.array([110.]),np.array([.2]))]
        np.testing.assert_allclose(overlay_spx(nav,cycles,entries,np.array([100.,100.,100.]),100.),nav)

    def test_spx_stays_invested_when_options_are_unavailable(self):
        nav = np.full((3,1),100.)
        cycles = [dict(ix=np.array([0,1])),dict(ix=np.array([1,2]))]
        entries = [(np.array([100.]),np.array([0.])),(np.array([100.]),np.array([0.]))]
        actual = overlay_spx(nav,cycles,entries,np.array([100.,90.,99.]),100.)
        np.testing.assert_allclose(actual[:,0],[100.,90.,99.])

    def test_missing_roll_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            overlay_spx(np.ones((2,1)),[dict(ix=np.array([0,1]))],[],np.ones(2))


if __name__ == "__main__":
    unittest.main()
