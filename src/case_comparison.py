from case_parser import CaseEntry
from case_tree import CaseTree
from case_crawler import CaseCrawler
from constants import HEADERS_AND_FORMAT, RUSSIAN_STOPWORDS, LOGS_DIRECTORY

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from logging import basicConfig, info, INFO
from datetime import datetime
from re import sub


def create_log_file():
    timestamp = sub('[: ]', '-', str(datetime.now())[:19])
    basicConfig(filename=LOGS_DIRECTORY + "comparison_" + timestamp + ".log", level=INFO, encoding="utf-8")


class CaseComparison:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    def tokenize_raw_data(self, case_entry: CaseEntry) -> list[str]:
        tokens = word_tokenize(case_entry.data)
        tokens = [token.lower() for token in tokens if len(token) > 2 and token.lower() not in RUSSIAN_STOPWORDS
                  and token.lower() not in HEADERS_AND_FORMAT]

        return tokens

    def tokenize_additional_info(self, case_entry: CaseEntry) -> list[str]:
        tokens = (" ".join([case_entry.description, case_entry.preconditions, " ".join(case_entry.contents),
                            " ".join(case_entry.expectations)])).split(" ")
        tokens = [token for token in tokens if len(token) > 2 and token not in RUSSIAN_STOPWORDS
                  and token not in HEADERS_AND_FORMAT]

        stemmer = SnowballStemmer("russian")
        tokens = [stemmer.stem(token) for token in tokens]

        return tokens

    def create_sentences_for_bert(self, tokens: list[str], sentence_length: int = 10) -> list[str]:
        return [" ".join(tokens[idx: idx + sentence_length]) for idx in range(0, len(tokens), sentence_length)]

    def vectorize_and_count_similarity(self, tokens1: list[str], tokens2: list[str]) -> list[list[float]]:
        vec = TfidfVectorizer()
        vector1 = vec.fit_transform(tokens1)
        vector2 = vec.transform(tokens2)

        similarity = cosine_similarity(vector1, vector2)

        return similarity

    def vectorize_with_bert_and_count_similarity(self, sentences1: list[str],
                                                 sentences2: list[str]) -> list[list[float]]:
        sentence_embeddings = self.model.encode(sentences1 + sentences2)
        return cosine_similarity(sentence_embeddings[:len(sentences1)], sentence_embeddings[len(sentences1):])

    def evaluate_cosine_similarity(self, similarity: list[list[float]], labels1: list[str], labels2: list[str],
                                   decay: float = 0.6, metric: str = "average", print_labels: bool = False) -> float:
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

    def compare(self, case_entry1: CaseEntry, case_entry2: CaseEntry, sentence_length: int = 10,
                decay: float = 0.6, metric: str = "average", mode: str = "silent") -> dict[str, float]:
        if mode == "silent" or mode == "print":
            if mode == "print":
                print("Comparing additional data of entries:")
            tokens1, tokens2 = self.tokenize_additional_info(case_entry1), self.tokenize_additional_info(case_entry2)
            cos_sim_tfidf = self.vectorize_and_count_similarity(tokens1, tokens2)
            score_tfidf = self.evaluate_cosine_similarity(cos_sim_tfidf, tokens1, tokens2, decay=decay, metric=metric,
                                                          print_labels=False)
            if mode == "print":
                print('Score on Tfidf: ', score_tfidf)
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
                print('Score on BERT: ', score_bert)
            return dict([("score_tfidf", score_tfidf), ("score_bert", score_bert)])

        elif mode == "log":
            info("Comparing additional data of entries:")
            info("Creating tokens for additional data")
            tokens1, tokens2 = self.tokenize_additional_info(case_entry1), self.tokenize_additional_info(case_entry2)
            info("done")
            info("Creating Tfidf vectors and counting their cosine similarity")
            cos_sim_tfidf = self.vectorize_and_count_similarity(tokens1, tokens2)
            info("done")
            info('Evaluating cosine similarity')
            score_tfidf = self.evaluate_cosine_similarity(cos_sim_tfidf, tokens1, tokens2, decay=decay, metric=metric,
                                                          print_labels=False)
            info('Score on Tfidf ' + str(score_tfidf))
            info("Splitting tokens for BERT sentence vectorization")
            sentences1, sentences2 = (self.create_sentences_for_bert(tokens1, sentence_length=sentence_length),
                                      self.create_sentences_for_bert(tokens2, sentence_length=sentence_length))
            info("Creating BERT vectors and counting their cosine similarity")
            cos_sim_bert = self.vectorize_with_bert_and_count_similarity(sentences1, sentences2)
            info("done")
            info('Evaluating cosine similarity')
            score_bert = self.evaluate_cosine_similarity(cos_sim_bert, sentences1, sentences2, decay=decay,
                                                         metric=metric,
                                                         print_labels=False)
            info('Score on BERT ' + str(score_bert))
            return dict([("score_tfidf", score_tfidf), ("score_bert", score_bert)])

    def compare_case_entry_with_tree(self, case_entry: CaseEntry, case_tree: CaseTree, sentence_length: int = 10,
                                     decay: float = 0.6, metric: str = "average", mode: str = "silent"):
        if mode == "silent" or mode == "print":
            if mode == "print":
                print("Comparing entry with '" + case_tree.case_entry.title + "' (id:" + str(
                    case_tree.case_entry.id) + ")")
            print(self.compare(case_entry, case_tree.case_entry, decay=decay, metric=metric,
                               sentence_length=sentence_length, mode=mode))
            if mode == "print":
                print("---------------------------------------")
            for sub_case_tree in case_tree.sub_case_trees:
                self.compare_case_entry_with_tree(case_entry, sub_case_tree, decay=decay, metric=metric,
                                                  sentence_length=sentence_length, mode=mode)

        elif mode == "log":
            if case_tree.level == 0:
                create_log_file()
                info("Comparing entry '" + case_entry.title + "' (id: " + str(case_entry.id) + ")")

            info("Comparing entry with '" + case_tree.case_entry.title + "' (id: " + str(case_tree.case_entry.id) + ")")
            info(self.compare(case_entry, case_tree.case_entry, decay=decay, metric=metric,
                              sentence_length=sentence_length, mode=mode))
            for sub_case_tree in case_tree.sub_case_trees:
                self.compare_case_entry_with_tree(case_entry, sub_case_tree, decay=decay, metric=metric,
                                                  sentence_length=sentence_length, mode=mode)

    def compare_case_entry_with_directory(self, case_entry: CaseEntry, case_crawler: CaseCrawler,
                                          sentence_length: int = 10, decay: float = 0.6, metric: str = "average",
                                          mode: str = "silent") -> list[CaseEntry]:
        possible_duplications = []
        print("Found " + str(len(case_crawler.case_entries)) + " cases in directory.\n")

        if mode == "silent":
            for i, ce in enumerate(case_crawler.case_entries):
                comp_result = self.compare(case_entry, ce, decay=decay, metric=metric,
                                           sentence_length=sentence_length, mode=mode)
                if comp_result["score_bert"] > 0:
                    possible_duplications.append(ce)
                if i > 0 and i % 9 == 0 or i == len(case_crawler.case_entries) - 1:
                    print("Proceeded " + str(i + 1) + " cases")

        elif mode == "print":
            print("Comparing entry '" + case_entry.title + "' (id:" + str(case_entry.id) + ")")
            print('\n---------------------------------------\n')
            for ce in case_crawler.case_entries:
                print("Comparing entry with '", ce.title, "' (id:", ce.id, ")")
                comp_result = self.compare(case_entry, ce, decay=decay, metric=metric,
                                           sentence_length=sentence_length, mode=mode)
                if comp_result["score_bert"] > 0:
                    possible_duplications.append(ce)
                print('\n---------------------------------------\n')

        elif mode == "log":
            create_log_file()
            info("Comparing entry '" + case_entry.title + "' (id: " + str(case_entry.id) + ")")
            for ce in case_crawler.case_entries:
                info("Comparing entry with '" + ce.title + "' (id: " + str(ce.id) + ")")
                comp_result = self.compare(case_entry, ce, decay=decay, metric=metric,
                                           sentence_length=sentence_length, mode=mode)
                if comp_result["score_bert"] > 0:
                    possible_duplications.append(ce)

        if len(possible_duplications) > 0:
            print("\nPossible duplications of case found:")
            for ce in possible_duplications:
                print(ce.title)
        else:
            print("\nNo duplications found.")
        return possible_duplications