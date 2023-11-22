import unittest

from case_parser import CaseEntry
from keyword_detection import KeywordDetection


class TestKeywordDetection(unittest.TestCase):
    def test_dummy_syntax_check(self):
        ce = CaseEntry("../../src/case-examples/synthetic_bad_keyword_reference.md")
        kd = KeywordDetection()
        result = kd.dummy_syntax_check(ce)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
