from __future__ import annotations

from pathlib import Path
import re
import tomllib
import unittest

import w90flow


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_package_version_matches_project_metadata(self) -> None:
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(w90flow.__version__, metadata["project"]["version"])

    def test_readme_commands_and_local_links_are_checkout_safe(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("运行W90Flow.command", readme)
        for target in re.findall(r"\[[^]]+\]\(([^)]+\.md)\)", readme):
            self.assertFalse(target.startswith(".."), target)
            self.assertTrue((ROOT / target).is_file(), target)


if __name__ == "__main__":
    unittest.main()
