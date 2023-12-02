import unittest

from case_crawler import CaseCrawler
from case_parser import CaseEntry
from keyword_detection import KeywordDetection


class TestKeywordDetection(unittest.TestCase):
    def __init__(self, method_name: str = ...):
        super().__init__(method_name)
        self.kd = KeywordDetection()

    def test_compiles(self):
        self.assertIsNotNone(self.kd)

    def test_dummy_syntax_check(self):
        ce = CaseEntry("../src/case-examples/synthetic_bad_keyword_reference.md")
        result = self.kd.dummy_syntax_check(ce)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_match_keywords_with_entry(self):
        ce = CaseEntry("../src/case-examples/synthetic_keyword_for_not_linked.md")
        keywords = ['Получены логин и пароль для входа в систему NNN.']
        result = self.kd.match_keywords_with_entry(keywords, ce, metric="absolute")
        self.assertIsNotNone(result)
        self.assertTrue(result > 0)

    def test_match_keywords_with_directory(self):
        ce = CaseCrawler("../src/case-examples")
        keywords = ['Получены логин и пароль для входа в систему NNN.']
        result = self.kd.match_keywords_with_directory(keywords, ce, metric="absolute", mode="silent")
        self.assertIsNotNone(result)
        self.assertTrue(len(result) > 0)


if __name__ == '__main__':
    unittest.main()
