from case_crawler import CaseCrawler
from case_parser import CaseEntry
from case_tree import CaseTree
from constants import HEADERS_AND_FORMAT, RUSSIAN_STOPWORDS, LOGS_DIRECTORY

from datetime import datetime
from logging import basicConfig, info, INFO

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk import ngrams
from rake_nltk import Rake

from re import sub

from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import Callable


def create_log_file():
    """
    Configure log file to write.
    """

    timestamp = sub('[: ]', '-', str(datetime.now())[:19])
    basicConfig(filename=LOGS_DIRECTORY + "comparison_" + timestamp + ".log", level=INFO, encoding="utf-8")


class CaseComparison:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        self.method_to_function = dict([("tfidf", self.compare_using_tfidf),
                                        ("random_sentences", self.compare_using_random_sentences),
                                        ("rake", self.compare_using_rake_phrases),
                                        ("ngrams", self.compare_using_ngrams)
                                        ])

    def tokenize_raw_data(self, case_entry: CaseEntry) -> list[str]:
        """
        Creates tokens from raw case data.
        :param case_entry: case to create tokens from.
        :return: list of tokens.
        """

        tokens = word_tokenize(case_entry.data)
        tokens = map(str.lower, tokens)
        tokens = [token for token in tokens if len(token) > 2 and token not in RUSSIAN_STOPWORDS
                  and token not in HEADERS_AND_FORMAT]

        return tokens

    def tokenize_additional_info(self, case_entry: CaseEntry) -> list[str]:
        """
        Creates tokens from description, preconditions and steps of case.
        Uses stemmer.
        :param case_entry: case to create tokens from.
        :return: list of tokens.
        """

        tokens = (" ".join([case_entry.description, case_entry.preconditions,
                            " ".join(case_entry.contents),
                            " ".join(case_entry.expectations)])).split(" ")
        tokens = [token for token in tokens if len(token) > 2 and token not in RUSSIAN_STOPWORDS
                  and token not in HEADERS_AND_FORMAT]

        stemmer = SnowballStemmer("russian")
        tokens = [stemmer.stem(token) for token in tokens]

        return tokens

    def create_rake_phrases(self, case_entry: CaseEntry) -> list[str]:
        """
        Creates sentences from raw case data using RAKE algorithm.
        :param case_entry: case to create sentences from.
        :return: list of sentences.
        """

        text = sub("\n+", " ", case_entry.data).strip()
        text = sub("[`*\"\']+", "", text)
        text = sub(r'[\\/|\-\[\]=+{}#~@№$%^&]+', " ", text)
        text = sub(r' +', ' ', text)
        rake = Rake()
        rake.extract_keywords_from_text(text)
        return rake.get_ranked_phrases()

    def create_ngrams(self, tokens: list[str], sentence_length: int = 10) -> list[str]:
        """
        Creates sentences from ngrams of given length.
        :param tokens: tokens to create ngrams from.
        :param sentence_length:
        :return: list of sentences.
        """

        len_grams = ngrams(tokens, sentence_length)
        return [" ".join(list(gram)) for gram in len_grams]

    def create_sentences_for_bert(self, tokens: list[str], sentence_length: int = 10) -> list[str]:
        """
        Creates sentences of given length using simple split.
        :param tokens: tokens to create ngrams from.
        :param sentence_length:
        :return: list of sentences.
        """

        return [" ".join(tokens[idx: idx + sentence_length]) for idx in range(0, len(tokens), sentence_length)]

    def vectorize_with_tfidf_and_count_similarity(self, tokens1: list[str], tokens2: list[str]) -> list[list[float]]:
        """
        Vectorize tokens of two cases with Tfidf and finds cosine similarity matrix.
        :param tokens1:
        :param tokens2:
        :return: cosine similarity matrix.
        """

        vec = TfidfVectorizer()
        vector1 = vec.fit_transform(tokens1)
        vector2 = vec.transform(tokens2)

        similarity = cosine_similarity(vector1, vector2)

        return similarity

    def vectorize_with_bert_and_count_similarity(self, sentences1: list[str],
                                                 sentences2: list[str]) -> list[list[float]]:
        """
        Vectorize sentences of two cases with BERT model and finds cosine similarity matrix.
        :param sentences1:
        :param sentences2:
        :return: cosine similarity matrix.
        """

        sentence_embeddings = self.model.encode(sentences1 + sentences2)
        return cosine_similarity(sentence_embeddings[:len(sentences1)], sentence_embeddings[len(sentences1):])

    def evaluate_cosine_similarity(self, similarity: list[list[float]], labels1: list[str], labels2: list[str],
                                   decay: float = 0.6, metric: str = "average", print_labels: bool = False) -> float:
        """
        Counts metric on given cosine similarity.
        Prints labels that match similarity decay if print_labels set True.
        :param similarity: cosine similarity matrix.
        :param labels1:
        :param labels2:
        :param decay: minimum value of cosine similarity that should affect metric.
        :param metric: either "average" or "absolute" value of cosine similarities that matched decay.
        :param print_labels: ``True`` if labels whose similarity matched decay to be print, or ``False`` if not.
        :return: value of metric.
        """

        count = 0
        for i in range(len(labels1)):
            for j in range(len(labels2)):
                if similarity[i][j] >= decay:
                    if print_labels:
                        print('"', labels1[i], '" + "', labels2[j], '" =', similarity[i][j])
                    count += 1
        if metric == "average":
            return count / (len(labels1) * len(labels2))
        if metric == "absolute":
            return count

    def evaluate_comparison_result(self, comp_result: dict[str, float], metric: str = "average") -> bool:
        """
        Decides if two cases are similar or not after comparison.
        :param comp_result: dict result of compare* function.
        :param metric: either "average" or "absolute" value of cosine similarities that matched decay.
        :return: ``True`` if cases are similar and ``False`` if not.
        """

        score_name = [*comp_result][0]
        if score_name == "score_tfidf":
            if metric == "average":
                return comp_result["score_tfidf"] >= 0.01
            if metric == "absolute":
                return comp_result["score_tfidf"] >= 100
        if score_name == "score_random_sentences":
            return comp_result["score_random_sentences"] > 0
        if score_name == "score_rake":
            return comp_result["score_rake"] > 0
        if score_name == "score_ngrams":
            if metric == "average":
                return comp_result["score_tfidf"] >= 0.05
            if metric == "absolute":
                return comp_result["score_tfidf"] >= 100

    def compare_using_ngrams(self, case_entry1: CaseEntry, case_entry2: CaseEntry,
                             sentence_length: int = 10, decay: float = 0.6, metric: str = "average",
                             mode: str = "silent") -> dict[str, float]:
        """
        Performs comparison using ngrams sentences.
        """

        if mode == "silent" or mode == "print":
            if mode == "print":
                print("Using ngrams to compare entries:")
            tokens1, tokens2 = self.tokenize_additional_info(case_entry1), self.tokenize_additional_info(case_entry2)
            sentences1, sentences2 = (self.create_ngrams(tokens1, sentence_length),
                                      self.create_ngrams(tokens2, sentence_length))
            cos_sim_bert = self.vectorize_with_bert_and_count_similarity(sentences1, sentences2)
            if mode == "print":
                score_ngrams = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                               metric=metric,
                                                               print_labels=True)
            else:
                score_ngrams = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                               metric=metric,
                                                               print_labels=False)
            if mode == "print":
                print('Score on BERT with ngrams: ', score_ngrams)
            return dict([("score_ngrams", score_ngrams)])

        elif mode == "log":
            info("Using ngrams to compare entries:")
            info("Creating tokens for additional data")
            tokens1, tokens2 = self.tokenize_additional_info(case_entry1), self.tokenize_additional_info(case_entry2)
            info("done")
            info("Creating ngrams for BERT sentence vectorization")
            sentences1, sentences2 = (self.create_ngrams(tokens1, sentence_length),
                                      self.create_ngrams(tokens2, sentence_length))
            info("done")
            info("Creating BERT vectors and counting their cosine similarity")
            cos_sim_bert = self.vectorize_with_bert_and_count_similarity(sentences1, sentences2)
            info("done")
            info('Evaluating cosine similarity')
            score_ngrams = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                           metric=metric,
                                                           print_labels=False)
            info('Score on BERT with ngrams ' + str(score_ngrams))
            return dict([("score_ngrams", score_ngrams)])

    def compare_using_rake_phrases(self, case_entry1: CaseEntry, case_entry2: CaseEntry,
                                   sentence_length: int = 10, decay: float = 0.6, metric: str = "average",
                                   mode: str = "silent") -> dict[str, float]:
        """
        Performs comparison using RAKE phrases.
        """

        if mode == "silent" or mode == "print":
            if mode == "print":
                print("Using Rake to compare entries:")
            sentences1, sentences2 = self.create_rake_phrases(case_entry1), self.create_rake_phrases(case_entry2)
            sentences1 = sentences1[:len(sentences1) // 2]
            sentences2 = sentences2[:len(sentences2) // 2]
            cos_sim_bert = self.vectorize_with_bert_and_count_similarity(sentences1, sentences2)
            if mode == "print":
                score_rake = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                             metric=metric,
                                                             print_labels=True)
            else:
                score_rake = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                             metric=metric,
                                                             print_labels=False)
            if mode == "print":
                print('Score on BERT with Rake: ', score_rake)
            return dict([("score_rake", score_rake)])

        elif mode == "log":
            info("Using Rake to compare entries:")
            info("Creating phrases for BERT sentence vectorization")
            sentences1, sentences2 = self.create_rake_phrases(case_entry1), self.create_rake_phrases(case_entry2)
            sentences1 = sentences1[:len(sentences1) // 2]
            sentences2 = sentences2[:len(sentences2) // 2]
            info("done")
            info("Creating BERT vectors and counting their cosine similarity")
            cos_sim_bert = self.vectorize_with_bert_and_count_similarity(sentences1, sentences2)
            info("done")
            info('Evaluating cosine similarity')
            score_rake = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                         metric=metric,
                                                         print_labels=False)
            info('Score on BERT with Rake ' + str(score_rake))
            return dict([("score_rake", score_rake)])

    def compare_using_tfidf(self, case_entry1: CaseEntry, case_entry2: CaseEntry, sentence_length: int = 10,
                            decay: float = 0.6, metric: str = "average", mode: str = "silent") -> dict[str, float]:
        """
        Performs comparison using tfidf vectorization.
        """

        if mode == "silent" or mode == "print":
            if mode == "print":
                print("Using Tfidf vectors to compare entries:")
            tokens1, tokens2 = self.tokenize_additional_info(case_entry1), self.tokenize_additional_info(case_entry2)
            cos_sim_tfidf = self.vectorize_with_tfidf_and_count_similarity(tokens1, tokens2)
            score_tfidf = self.evaluate_cosine_similarity(cos_sim_tfidf, tokens1, tokens2, decay=decay, metric=metric,
                                                          print_labels=False)
            if mode == "print":
                print('Score on Tfidf: ', score_tfidf)
            return dict([("score_tfidf", score_tfidf)])

        elif mode == "log":
            info("Using Tfidf vectors to compare entries:")
            info("Creating tokens for additional data")
            tokens1, tokens2 = self.tokenize_additional_info(case_entry1), self.tokenize_additional_info(case_entry2)
            info("done")
            info("Creating Tfidf vectors and counting their cosine similarity")
            cos_sim_tfidf = self.vectorize_with_tfidf_and_count_similarity(tokens1, tokens2)
            info("done")
            info('Evaluating cosine similarity')
            score_tfidf = self.evaluate_cosine_similarity(cos_sim_tfidf, tokens1, tokens2, decay=decay, metric=metric,
                                                          print_labels=False)
            info('Score on Tfidf ' + str(score_tfidf))
            return dict([("score_tfidf", score_tfidf)])

    def compare_using_random_sentences(self, case_entry1: CaseEntry, case_entry2: CaseEntry, sentence_length: int = 10,
                                       decay: float = 0.6, metric: str = "average",
                                       mode: str = "silent") -> dict[str, float]:
        """
        Performs comparison using simple sentences.
        """
        if mode == "silent" or mode == "print":
            if mode == "print":
                print("Using random sentence split to compare entries:")
            tokens1, tokens2 = self.tokenize_additional_info(case_entry1), self.tokenize_additional_info(case_entry2)
            sentences1, sentences2 = (self.create_sentences_for_bert(tokens1, sentence_length=sentence_length),
                                      self.create_sentences_for_bert(tokens2, sentence_length=sentence_length))
            cos_sim_bert = self.vectorize_with_bert_and_count_similarity(sentences1, sentences2)
            if mode == "print":
                score_bert = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                             metric=metric,
                                                             print_labels=True)
            else:
                score_bert = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                             metric=metric,
                                                             print_labels=False)
            if mode == "print":
                print('Score on random sentences: ', score_bert)
            return dict([("score_random_sentences", score_bert)])

        elif mode == "log":
            info("Using random sentence split to compare entries:")
            info("Creating tokens for additional data")
            tokens1, tokens2 = self.tokenize_additional_info(case_entry1), self.tokenize_additional_info(case_entry2)
            info("done")
            info("Splitting tokens for BERT sentence vectorization")
            sentences1, sentences2 = (self.create_sentences_for_bert(tokens1, sentence_length=sentence_length),
                                      self.create_sentences_for_bert(tokens2, sentence_length=sentence_length))
            info("done")
            info("Creating BERT vectors and counting their cosine similarity")
            cos_sim_bert = self.vectorize_with_bert_and_count_similarity(sentences1, sentences2)
            info("done")
            info('Evaluating cosine similarity')
            score_bert = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                         metric=metric,
                                                         print_labels=False)
            info('Score on random sentences ' + str(score_bert))
            return dict([("score_random_sentences", score_bert)])

    def compare(self, case_entry1: CaseEntry, case_entry2: CaseEntry, sentence_length: int = 10,
                decay: float = 0.6, metric: str = "average", mode: str = "silent",
                method: str = "random_sentences") -> dict[str, float]:
        """
        Compares two cases and count a similarity metric.
        :param case_entry1:
        :param case_entry2:
        :param sentence_length:
        :param decay: minimum value of cosine similarity that should affect metric.
        :param metric: either "average" or "absolute" value of cosine similarities that matched decay.
        :param mode: either "silent", "print" if labels to be print, or "log" to write in .log file.
        :param method: name of comparison method.
        :return: dict[method : score]
        """
        comparator: Callable = self.method_to_function[method]
        return comparator(case_entry1=case_entry1, case_entry2=case_entry2, sentence_length=sentence_length,
                          decay=decay, metric=metric, mode=mode)

    def compare_case_entry_with_tree(self, case_entry: CaseEntry, case_tree: CaseTree, sentence_length: int = 10,
                                     decay: float = 0.6, metric: str = "average", mode: str = "silent",
                                     method: str = "random_sentences"):
        """
        Compares case with case tree, performs comparison for given case and each case in tree.
        """

        if mode == "silent" or mode == "print":
            if mode == "print":
                print("Comparing entry with '" + case_tree.case_entry.title + "' (id:" + str(
                    case_tree.case_entry.id) + ")")
            print(self.compare(case_entry, case_tree.case_entry, decay=decay, metric=metric,
                               sentence_length=sentence_length, mode=mode, method=method))
            if mode == "print":
                print("---------------------------------------")
            for sub_case_tree in case_tree.sub_case_trees:
                self.compare_case_entry_with_tree(case_entry, sub_case_tree, decay=decay, metric=metric,
                                                  sentence_length=sentence_length, mode=mode, method=method)

        elif mode == "log":
            if case_tree.level == 0:
                create_log_file()
                info("Comparing entry '" + case_entry.title + "' (id: " + str(case_entry.id) + ")")

            info("Comparing entry with '" + case_tree.case_entry.title + "' (id: " + str(case_tree.case_entry.id) + ")")
            info(self.compare(case_entry, case_tree.case_entry, decay=decay, metric=metric,
                              sentence_length=sentence_length, mode=mode, method=method))
            for sub_case_tree in case_tree.sub_case_trees:
                self.compare_case_entry_with_tree(case_entry, sub_case_tree, decay=decay, metric=metric,
                                                  sentence_length=sentence_length, mode=mode, method=method)

    def compare_case_entry_with_directory(self, case_entry: CaseEntry, case_crawler: CaseCrawler,
                                          sentence_length: int = 10, decay: float = 0.6, metric: str = "average",
                                          mode: str = "silent", method: str = "random_sentences") -> list[CaseEntry]:
        """
        Compares case with cases of directory, performs comparison for given case and each case of directory.
        """

        possible_duplications = []
        print("Found " + str(len(case_crawler.case_entries)) + " cases in directory.\n")

        if mode == "silent":
            for i, ce in enumerate(case_crawler.case_entries):
                comp_result = self.compare(case_entry, ce, decay=decay, metric=metric,
                                           sentence_length=sentence_length, mode=mode, method=method)
                if self.evaluate_comparison_result(comp_result=comp_result, metric=metric):
                    possible_duplications.append(ce)
                if i > 0 and i % 9 == 0 or i == len(case_crawler.case_entries) - 1:
                    print("Proceeded " + str(i + 1) + " cases")

        elif mode == "print":
            print("Comparing entry '" + case_entry.title + "' (id:" + str(case_entry.id) + ")")
            print('\n---------------------------------------\n')
            for ce in case_crawler.case_entries:
                print("Comparing entry with '", ce.title, "' (id:", ce.id, ")")
                comp_result = self.compare(case_entry, ce, decay=decay, metric=metric,
                                           sentence_length=sentence_length, mode=mode, method=method)
                if self.evaluate_comparison_result(comp_result=comp_result, metric=metric):
                    possible_duplications.append(ce)
                print('\n---------------------------------------\n')

        elif mode == "log":
            create_log_file()
            info("Comparing entry '" + case_entry.title + "' (id: " + str(case_entry.id) + ")")
            for ce in case_crawler.case_entries:
                info("Comparing entry with '" + ce.title + "' (id: " + str(ce.id) + ")")
                comp_result = self.compare(case_entry, ce, decay=decay, metric=metric,
                                           sentence_length=sentence_length, mode=mode, method=method)
                if self.evaluate_comparison_result(comp_result=comp_result, metric=metric):
                    possible_duplications.append(ce)

        if len(possible_duplications) > 0:
            print("\nPossible duplications of case found:")
            for ce in possible_duplications:
                print(ce.title + " (" + ce.path + ")")
        else:
            print("\nNo duplications found.")
        return possible_duplications
