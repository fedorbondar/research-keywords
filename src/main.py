from case_parser import CaseEntry
from case_comparison import CaseComparison
from case_crawler import CaseCrawler

from sys import argv

if __name__ == "__main__":
    if len(argv) < 3 or len(argv) > 4:
        print("Usage: python main.py folder_with_cases new_case.md [silent | print | log]")
        exit(0)

    crawler = CaseCrawler(argv[1])
    new_case = CaseEntry(argv[2])

    mode = "silent"
    if len(argv) == 4:
        mode = argv[3]

    comp = CaseComparison()
    res = comp.compare_case_entry_with_directory(new_case, crawler, sentence_length=7, decay=0.9, metric="absolute",
                                                 mode=mode, method="random_sentences")
