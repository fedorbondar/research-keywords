from case_parser import CaseEntry
from case_comparison import CaseComparison


class KeywordDetection:
    def dummy_syntax_check(self, case_entry: CaseEntry) -> list[str]:
        cc = CaseComparison()
        tokens = cc.tokenize_additional_info(case_entry)
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
