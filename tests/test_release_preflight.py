from __future__ import annotations

from pathlib import Path
import sys
import unittest


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import release_preflight


class ReleasePreflightTests(unittest.TestCase):
    def test_current_repository_passes_offline_release_preflight(self):
        result = release_preflight.run_preflight(Path(__file__).resolve().parents[1])
        self.assertTrue(result["passed"], result["failures"])
        self.assertGreater(result["scanned_text_files"], 10)
        # Secret files (.env, minimax api.txt) are present locally but must
        # only generate warnings, never failures.
        warning_codes = {item["code"] for item in result["warnings"]}
        self.assertIn("ignored_local_secret_file_present", warning_codes)
        # No tracked secret risks must be staged for commit.
        self.assertEqual(result["failures"], [])


if __name__ == "__main__":
    unittest.main()
