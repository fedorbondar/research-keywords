from case_comparison import CaseComparison
from case_crawler import CaseCrawler
from case_parser import CaseEntry

from re import findall


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

    def get_possible_keywords_from_preconditions(self, case_entry: CaseEntry, sentence_length: int = 10) -> list[str]:
        """
        Finds possible keyword phrases in preconditions section
        :param case_entry: case entry to check
        :param sentence_length: length of phrases to obtain
        :return: list of keyword phrases
        """

        tokens = case_entry.preconditions.split(" ")
        return [" ".join(tokens[idx: idx + sentence_length]) for idx in range(0, len(tokens), sentence_length)]

    def match_keywords_with_entry_exactly(self, keyword_phrases: list[str], case_entry: CaseEntry) -> int:
        """
        Finds matches of keyword phrases in case entry using exact matching
        :param keyword_phrases: keyword phrases to match
        :param case_entry: case entry to check
        :return: number of matches
        """

        occurrences = 0
        data = case_entry.data.lower()
        for phrase in keyword_phrases:
            occurrences += len(findall(phrase.lower(), data))

        return occurrences

    def match_keywords_with_entry(self, keyword_phrases: list[str], case_entry: CaseEntry, sentence_length: int = 10,
                                  decay: float = 0.6, metric: str = "average", mode: str = "silent") -> float:
        """
        Finds matches of keyword phrases in case entry using BERT encoding
        :param keyword_phrases: keyword phrases to match
        :param case_entry: case entry to check
        :param sentence_length: length of sentences to match with keyword phrases
        :param decay: minimum value of cosine similarity that should affect metric.
        :param metric: either "average" or "absolute" value of cosine similarities that matched decay.
        :param mode: either "silent" or "print" if labels to be print.
        :return: metric of occurrence
        """

        tokens = self.cc.tokenize_raw_data(case_entry)
        sentences = self.cc.create_sentences_for_bert(tokens, sentence_length=sentence_length)
        cos_sim_matrix = self.cc.vectorize_with_bert_and_count_similarity(keyword_phrases, sentences)
        occurrence = self.cc.evaluate_cosine_similarity(cos_sim_matrix, keyword_phrases, sentences,
                                                        decay=decay, metric=metric, print_labels=(mode == "print"))
        return occurrence

    def match_keywords_with_directory(self, keyword_phrases: list[str], case_crawler: CaseCrawler,
                                      sentence_length: int = 10, decay: float = 0.6, metric: str = "average",
                                      mode: str = "silent") -> list[CaseEntry]:
        """
        Finds matches of keyword phrases in directory
        :param keyword_phrases: keyword phrases to match
        :param case_crawler: crawler of directory with case entries
        :param sentence_length: length of sentences to match with keyword phrases
        :param decay: minimum value of cosine similarity that should affect metric.
        :param metric: either "average" or "absolute" value of cosine similarities that matched decay.
        :param mode: either "silent" or "print" if labels to be print.
        :return: list of case entries that matched with keyword phrases
        """

        matches = []
        if mode == "print":
            print("\nFound " + str(len(case_crawler.case_entries)) + " cases in directory.")

        for entry in case_crawler.case_entries:
            occurrence = self.match_keywords_with_entry(keyword_phrases, entry, sentence_length, decay, metric, mode)
            if occurrence > 0:
                matches.append(entry)

        if mode == "print":
            if len(matches) == 0:
                print("No matches found.")
            else:
                print("\nFound " + str(len(matches)) + " matches in directory:")
                for entry in matches:
                    print(entry.title + " (" + entry.path + ")")
        return matches
