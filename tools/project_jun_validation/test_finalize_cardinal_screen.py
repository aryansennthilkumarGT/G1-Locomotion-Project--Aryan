"""Final aggregation must never turn missing evidence into a completed benchmark."""

import unittest
from pathlib import Path
from unittest.mock import patch

import finalize_cardinal_screen as final


class FinalScreenTests(unittest.TestCase):
    def test_all_missing(self):
        with patch.object(final, "verify", side_effect=FileNotFoundError("missing trace")):
            result = final.collect(Path("/fixture"))
        self.assertFalse(result["benchmark_complete"])
        self.assertEqual(len(result["verification_failures"]), 4)
        self.assertEqual(len(result["missing_headings"]), 4)


if __name__ == "__main__":
    unittest.main()
