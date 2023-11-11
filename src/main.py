from case_parser import CaseEntry
from case_comparison import CaseComparison
from case_crawler import CaseCrawler

from sys import argv

if __name__ == "__main__":
    crawler = CaseCrawler(argv[1])
    new_case = CaseEntry(argv[2])
    comp = CaseComparison()
    res = comp.compare_case_entry_with_directory(new_case, crawler,
                                                 sentence_length=7, decay=0.9, metric="absolute", mode="silent")
    # python case_crawler.py folder_with_cases new_case.md
