import unittest

from case_parser import CaseEntry


class TestParser(unittest.TestCase):
    def test_compiles(self):
        te = CaseEntry("../src/case-examples/synthetic_base.md")
        self.assertIsNotNone(te)

    def test_field_initialization_main_info(self):
        te = CaseEntry("../src/case-examples/synthetic_keyword1.md")
        self.assertIsNotNone(te.title)
        self.assertIsNotNone(te.id)
        self.assertIsNotNone(te.section_id)
        self.assertIsNotNone(te.url)
        self.assertIsNotNone(te.name_in_tors)
        self.assertIsNotNone(te.main_product)
        self.assertIsNotNone(te.refs)

    def test_contents_and_expectations_size_match(self):
        te = CaseEntry("../src/case-examples/synthetic_keyword2.md")
        self.assertEqual(len(te.contents), len(te.expectations))

    def test_keyword_is_matched_as_keyword(self):
        te = CaseEntry("../src/case-examples/synthetic_keyword3.md")
        self.assertTrue(te.is_keyword)


if __name__ == '__main__':
    unittest.main()
