import unittest

from case_parser import CaseEntry
from keyword_detection import KeywordDetection


class TestKeywordDetection(unittest.TestCase):
    def __init__(self, method_name: str = ...):
        super().__init__(method_name)
        self.kd = KeywordDetection()

    def test_compiles(self):
        self.assertIsNotNone(self.kd)

    def test_dummy_syntax_check(self):
        ce = CaseEntry("../../src/case-examples/synthetic_bad_keyword_reference.md")
        result = self.kd.dummy_syntax_check(ce)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)


if __name__ == '__main__':
    unittest.main()
