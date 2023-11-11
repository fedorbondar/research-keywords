from os import path, walk

from case_parser import CaseEntry


class CaseCrawler:
    def __init__(self, directory: str):
        self.directory: str = directory
        self.case_entries: list[CaseEntry] = []
        self.__find_markdown_in_sub_directories()

    def __find_markdown_in_sub_directories(self):
        for dir_path, dir_names, filenames in walk(self.directory):
            for filename in [f for f in filenames if f.endswith(".md")]:
                self.case_entries.append(CaseEntry(path.join(dir_path, filename)))

