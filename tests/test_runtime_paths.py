from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tag.config import load_config
from tag.runtime_paths import RuntimePathResolver


class RuntimePathResolverTests(unittest.TestCase):
    def test_store_override_keeps_store_scoped_logs_together(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {}, clear=True):
                paths = RuntimePathResolver(config=load_config()).resolve(store_dir_override=Path(td))
                self.assertEqual(paths.store_dir, Path(td).resolve())
                self.assertEqual(paths.dead_letter_path, Path(td).resolve() / "dead_letters.jsonl")
                self.assertEqual(paths.bus_event_log_path, Path(td).resolve() / "bus_events.log")

    def test_path_environment_application_overrides_previous_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            resolver = RuntimePathResolver(config=load_config())
            with mock.patch.dict(os.environ, {}, clear=True):
                resolver.resolve(store_dir_override=Path(first)).apply_environment(override=True)
                resolver.resolve(store_dir_override=Path(second)).apply_environment(override=True)
                self.assertEqual(os.environ["AXIOM_STORE_DIR"], str(Path(second).resolve()))
                self.assertEqual(os.environ["AXIOM_DEAD_LETTER_PATH"], str(Path(second).resolve() / "dead_letters.jsonl"))

    def test_native_candidates_follow_release_layout(self) -> None:
        paths = RuntimePathResolver(config=load_config()).resolve()
        candidates = [str(path) for path in paths.native_library_candidates(system="Linux")]
        joined = "\n".join(candidates)
        self.assertIn("Releases-x64/axi.so", joined)
        self.assertIn("compiled/binaries/Linux64/axirt.so", joined)
        self.assertNotIn("Releases-x64/axirt.so", joined)


if __name__ == "__main__":
    unittest.main()
