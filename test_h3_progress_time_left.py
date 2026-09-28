"""The H3 Now card says how long is LEFT, not a number that reads as a total.

REPORTED (Pinokio, 2026-09-20, a 48 GB Mac): a 15 s Draft showed
"27m 23s / ~18m 21s" in window 2 of 3, and the user read ~18 min as what the
whole render was supposed to take. It was the time left: the H3 run loop put
its remaining estimate in `eta_sec`, and the Now card prints `eta_sec` as
"<elapsed> / ~<eta>". The card already has a remaining-time form
("<elapsed> in · ~<left> left") for `remaining_sec`; the H3 loop now fills it.
The same estimate also left out the staged load / prompt encode every later
window pays before its first step (H3_LOAD_SEC in the cost model).

A text-level check: the loop is a subprocess reader with no seam to drive it,
and the card logic is a few lines in queue.js's poll renderer.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PANEL = (ROOT / "mlx_ltx_panel.py").read_text(encoding="utf-8")
QUEUE = (ROOT / "webapp" / "js" / "queue.js").read_text(encoding="utf-8")


class H3ProgressSaysTimeLeft(unittest.TestCase):
    def _h3_progress_block(self) -> str:
        i = PANEL.index('"window_total": tot_windows')
        return PANEL[i - 2500:i]

    def test_h3_progress_carries_remaining_sec(self):
        self.assertRegex(self._h3_progress_block(),
                         r'"remaining_sec":\s*eta if last_step else None')

    def test_later_windows_pay_their_loads(self):
        block = self._h3_progress_block()
        self.assertIn("_load_est * later", block)
        self.assertRegex(PANEL, r"_load_est = H3_LOAD_SEC \* _h3_speed_factor\(chain_windows\)")

    def test_now_card_prefers_the_remaining_form(self):
        i = QUEUE.index("prog.remaining_sec != null")
        j = QUEUE.index("prog.eta_sec", i)
        self.assertLess(i, j, "the card must try remaining_sec before eta_sec")
        self.assertIn("left`", QUEUE[i:j])


if __name__ == "__main__":
    unittest.main()
