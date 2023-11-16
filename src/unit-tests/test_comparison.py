import unittest

from case_parser import CaseEntry
from case_tree import CaseTree
from case_comparison import CaseComparison


class TestComparison(unittest.TestCase):
    def test_compiles(self):
        tc = CaseComparison()
        self.assertIsNotNone(tc)

    def test_dimensions_match_after_tfidf(self):
        tc = CaseComparison()
        te1 = CaseEntry("../../src/case-examples/synthetic_keyword1.md")
        te2 = CaseEntry("../../src/case-examples/synthetic_keyword2.md")
        t1, t2 = tc.tokenize_additional_info(te1), tc.tokenize_additional_info(te2)
        r = tc.vectorize_and_count_similarity(t1, t2)
        self.assertEqual(len(r), len(t1))
        self.assertEqual(len(r[0]), len(t2))

    def test_dimensions_match_after_bert(self):
        tc = CaseComparison()
        te1 = CaseEntry("../../src/case-examples/synthetic_keyword2.md")
        te2 = CaseEntry("../../src/case-examples/synthetic_keyword3.md")
        t1, t2 = tc.tokenize_additional_info(te1), tc.tokenize_additional_info(te2)
        s1 = tc.create_sentences_for_bert(t1)
        s2 = tc.create_sentences_for_bert(t2)
        r = tc.vectorize_with_bert_and_count_similarity(s1, s2)
        self.assertEqual(len(r), len(s1))
        self.assertEqual(len(r[0]), len(s2))

    def test_compare(self):
        tc = CaseComparison()
        te1 = CaseEntry("../../src/case-examples/synthetic_base.md")
        te2 = CaseEntry("../../src/case-examples/synthetic_keyword3.md")
        result = tc.compare(te1, te2, decay=0.6, metric="absolute", sentence_length=5, mode="silent",
                            method="random_sentences")
        self.assertTrue(result["score_random_sentences"] >= 0)

    def test_compare_with_tree(self):
        tt = CaseTree(CaseEntry("../../src/case-examples/synthetic_base.md"))
        te3 = CaseEntry("../../src/case-examples/synthetic_keyword3.md")
        tc = CaseComparison()
        tc.compare_case_entry_with_tree(te3, tt, decay=0.5, metric="average", sentence_length=5, mode="silent",
                                        method="random_sentences")


if __name__ == '__main__':
    unittest.main()
