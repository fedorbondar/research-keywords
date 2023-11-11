import unittest

from case_tree import CaseTree
from case_parser import CaseEntry


class TestTree(unittest.TestCase):
    def test_compiles(self):
        tt = CaseTree(CaseEntry("../../src/case-examples/synthetic_base.md"))
        self.assertIsNotNone(tt)

    def test_tree_builds_correctly(self):
        tt = CaseTree(CaseEntry("../../src/case-examples/synthetic_base.md"))
        kw1 = CaseEntry("../../src/case-examples/synthetic_keyword1.md")
        kw2 = CaseEntry("../../src/case-examples/synthetic_keyword2.md")
        self.assertEqual(2, len(tt.sub_case_trees))
        self.assertEqual(kw1.data, tt.sub_case_trees[0].case_entry.data)
        self.assertEqual(kw2.data, tt.sub_case_trees[1].case_entry.data)


if __name__ == '__main__':
    unittest.main()
