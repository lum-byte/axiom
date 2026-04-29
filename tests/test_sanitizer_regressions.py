from __future__ import annotations

import unittest

from tag.topology.sanitize import Sanitizer


class SanitizerRegressionTests(unittest.TestCase):
    def test_utf7_surrogate_sequence_does_not_crash(self) -> None:
        sanitizer = Sanitizer()
        raw = b"<html><body>safe +3rI- text</body></html>"
        result = sanitizer.process(raw)
        self.assertTrue(result.ok)
        self.assertIsInstance(result.text, str)


if __name__ == "__main__":
    unittest.main()
