from dataclasses import dataclass
from io import open
from re import findall, split, sub


@dataclass
class Keyword:
    keyword_content: str
    link_to_case: str
    link_to_keyword: str


@dataclass
class CaseEntry:
    path: str
    data: str
    sections: list[str]
    title: str
    name_in_tors: str
    id: int
    url: str
    section_id: int
    refs: str
    main_product: str
    description: str
    preconditions: str
    is_keyword: bool = False

    def __init__(self, path: str):
        self.path = path
        self.contents: list[str] = []
        self.expectations: list[str] = []
        self.keywords: list[Keyword] = []

        self.__load_data()
        self.__parse_sections()
        self.__extract_keywords()
        self.__create_main_info()
        self.__cleanup_punctuation()
        self.__create_additional_info()

    def __load_data(self):
        with open(self.path, mode="r", encoding="utf-8") as f:
            self.data = f.read()

    def __parse_sections(self):
        """
        Splits case description into sections by headings.
        """

        sections = sub("\n+", " ", self.data).strip()
        sections = sub("[`*\"\']+", "", sections)
        self.sections = split("#{1,2}", sections)[1:]

    def __extract_keywords(self):
        """
        Extracts keywords using following pattern matching:

        keyword = [``title``](``link to case``) [Link to keyword](``link to .md``)
        """

        for section in self.sections:
            content_and_case = findall(r'keyword = \[[^.]+]\(\S+\)', section)
            link_to_keyword = findall(r'\[Link to keyword]\(\S+\)', section)
            for i in range(len(content_and_case)):
                content = content_and_case[i][content_and_case[i].find('[') + 1: content_and_case[i].rfind(']')]
                case = content_and_case[i][content_and_case[i].rfind('(') + 1: content_and_case[i].rfind(')')]
                link = link_to_keyword[i][link_to_keyword[i].rfind('(') + 1: link_to_keyword[i].rfind(')')]
                self.keywords.append(Keyword(content, case, link))
        for i in range(len(self.sections)):
            self.sections[i] = sub(r'\[Link to keyword]\(\S+\)', "", self.sections[i])

    def __create_main_info(self):
        """
        Initializes main information of case: title, id, etc.
        """

        for section in self.sections:
            if section.startswith(" URL:"):
                self.url = sub("URL:", " ", section).strip()
            elif section.startswith(" Title:"):
                self.title = sub("Title:", " ", section).strip()
                if self.title.startswith("[Keyword]"):
                    self.is_keyword = True
            elif section.startswith(" Name in TORS:"):
                self.name_in_tors = sub("Name in TORS:", " ", section).strip()
            elif section.startswith(" Section ID:"):
                self.section_id = int(sub("Section ID:", " ", section).strip())
            elif section.startswith(" ID:"):
                self.id = int(sub("ID:", " ", section).strip())
            elif section.startswith(" Refs:"):
                self.refs = sub("Refs:", " ", section).strip()
            elif section.startswith(" Main Product:"):
                self.main_product = sub("Main Product:", " ", section).strip()

    def __cleanup_punctuation(self):
        """
        Deletes special symbols from rest of description.
        """

        for i in range(len(self.sections)):
            self.sections[i] = sub(r'[\\/:.!?,|\-()\[\]=+{}#~@№$;%^&]+', " ", self.sections[i])
            self.sections[i] = sub(r' +', ' ', self.sections[i])

    def __create_additional_info(self):
        """
        Initializes additional information of case: description, preconditions, steps (contents and expectations).
        """

        for section in self.sections:
            if section.startswith(" Description"):
                self.description = sub("Description", " ", section).strip().lower()
            elif section.startswith(" Preconditions"):
                self.preconditions = sub("Preconditions", " ", section).strip().lower()
            elif section.startswith(" Content"):
                self.contents.append(sub("Content", " ", section).strip().lower())
            elif section.startswith(" Expected"):
                self.expectations.append(sub("Expected", " ", section).strip().lower())
