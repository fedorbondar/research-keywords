import unittest

from case_parser import CaseEntry
from case_tree import CaseTree
from case_comparison import CaseComparison


class TestComparison(unittest.TestCase):
    def __init__(self, method_name: str = ...):
        super().__init__(method_name)
        self.tc = CaseComparison()

    def test_compiles(self):
        self.assertIsNotNone(self.tc)

    def test_dimensions_match_after_tfidf(self):
        te1 = CaseEntry("../../src/case-examples/synthetic_keyword1.md")
        te2 = CaseEntry("../../src/case-examples/synthetic_keyword2.md")
        t1, t2 = self.tc.tokenize_additional_info(te1), self.tc.tokenize_additional_info(te2)
        r = self.tc.vectorize_with_tfidf_and_count_similarity(t1, t2)
        self.assertEqual(len(r), len(t1))
        self.assertEqual(len(r[0]), len(t2))

    def test_dimensions_match_after_bert(self):
        te1 = CaseEntry("../../src/case-examples/synthetic_keyword2.md")
        te2 = CaseEntry("../../src/case-examples/synthetic_keyword3.md")
        t1, t2 = self.tc.tokenize_additional_info(te1), self.tc.tokenize_additional_info(te2)
        s1 = self.tc.create_sentences_for_bert(t1)
        s2 = self.tc.create_sentences_for_bert(t2)
        r = self.tc.vectorize_with_bert_and_count_similarity(s1, s2)
        self.assertEqual(len(r), len(s1))
        self.assertEqual(len(r[0]), len(s2))

    def test_compare(self):
        te1 = CaseEntry("../../src/case-examples/synthetic_base.md")
        te2 = CaseEntry("../../src/case-examples/synthetic_keyword3.md")
        result = self.tc.compare(te1, te2, decay=0.6, metric="absolute", sentence_length=5, mode="silent",
                                 method="random_sentences")
        self.assertTrue(result["score_random_sentences"] >= 0)

    def test_compare_with_tree(self):
        tt = CaseTree(CaseEntry("../../src/case-examples/synthetic_base.md"))
        te3 = CaseEntry("../../src/case-examples/synthetic_keyword3.md")
        self.tc.compare_case_entry_with_tree(te3, tt, decay=0.5, metric="average", sentence_length=5, mode="silent",
                                             method="random_sentences")


if __name__ == '__main__':
    unittest.main()
