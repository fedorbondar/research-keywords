from case_parser import CaseEntry


class CaseTree:
    """
    Data structure for cases and their keywords, also loaded as cases.
    """

    def __init__(self, case_entry: CaseEntry, level: int = 0):
        self.level = level
        self.case_entry = case_entry
        self.sub_case_trees: list[CaseTree] = []
        self.__create_sub_case_trees()

    def __create_sub_case_trees(self):
        """
        Builds tree recursively, keywords of case are subtrees.
        """

        for keyword in self.case_entry.keywords:
            self.sub_case_trees.append(CaseTree(CaseEntry(keyword.link_to_keyword), level=self.level+1))
