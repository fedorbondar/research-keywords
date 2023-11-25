from case_comparison import CaseComparison
from case_parser import CaseEntry


class KeywordDetection:
    def __init__(self):
        self.cc = CaseComparison()

    def dummy_syntax_check(self, case_entry: CaseEntry) -> list[str]:
        """
        Finds possible inappropriate mentions of keywords.
        :param case_entry: case to check.
        :return: list of possible keyword IDs.
        """

        tokens = self.cc.tokenize_additional_info(case_entry)
        suspicious = [token for token in tokens if token.isdigit() and (len(token) == 6 or len(token) == 7)]

        result = []
        for s in suspicious:
            found = False
            for keyword in case_entry.keywords:
                if keyword.link_to_case.rfind(s) != -1:
                    found = True
            if not found:
                result.append(s)

        return result
