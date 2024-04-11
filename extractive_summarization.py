import re
from collections import Counter
from typing import List, Dict

import numpy as np
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from rouge_score import rouge_scorer


class TextSummariser:
    """
    This class represent text summary generator for list of documents. A document consists multiple sentences and
    each sentence contains multiple words.
    """

    def __init__(self, doc: List[str]) -> None:
        """
        Initialise the Summariser object.
        :param doc: list of documents
        """
        self.doc = doc
        self.pdist = None

    def clean_text(self) -> List[str]:
        """
        Removes special chars, split sentences by "." and remove stop words from the document.
        :return: list of words present in the document
        """
        # removing special chars in document
        self.doc = [re.sub(r'[^a-zA-Z0-9.\s]', '', doc) for doc in self.doc]
        # removing stop words
        stopWords = set(stopwords.words('english'))
        filtered_words = [word for word in " ".join(self.doc).lower().split(" ") if word not in stopWords]
        sentences = " ".join(filtered_words).split(".")
        filtered_words = [item for item in filtered_words if "." not in item or item != '.']
        # setting word occurrence prob in document
        self.pdist = self.compute_probabilities(filtered_words)
        return sentences

    @staticmethod
    def compute_probabilities(words: List[str]) -> Dict[str, float]:
        """
        Compute the probability of occurrence of each word in the document.
        :param words: list of preprocessed words
        :return: dictionary with each word as key and values as their corresponding probability
        """
        word_freq = Counter(words)
        total_words = len(words)
        word_freq = {k: v / total_words for k, v in word_freq.items()}
        return word_freq

    def compute_kl(self, qdist: Dict[str, float]) -> float:
        """
        Generate summary text of original document using KL divergence.
        :param qdist: list of all words and their occurrence prob in sentence
        :return distance of sentence from document
        """
        kl_dist = 0
        for word, prob in qdist.items():
            kl_dist += (self.pdist.get(word, 0) * np.log(self.pdist.get(word, 0) / (prob + 0.01)))
        return abs(kl_dist)

    def generate_summary(self, n_sentences: int, sentences: List[str]) -> str:
        """
        Generate n summary sentences that are closely related to original document.
        :param sentences: List of sentences present in the document
        :param n_sentences: number of sentences
        :return: summary text
        """
        summary_sentence = ""
        sentences = set(sentences)
        while n_sentences > 0:
            max_sentence, min_score = "", float('inf')
            for sentence in sentences:
                if sentence:
                    sum_sentence = summary_sentence + sentence.strip()
                    qdist = self.compute_probabilities(sum_sentence.split(" "))
                    score = self.compute_kl(qdist)
                    if abs(score) < min_score:
                        max_sentence = sentence
                        min_score = score
            if max_sentence:
                sentences.remove(max_sentence)
            summary_sentence += max_sentence
            n_sentences -= 1
        return summary_sentence.strip()


if __name__ == "__main__":
    # duc_dataset = load_dataset("midas/duc2001")
    # duc_docs = []
    # for i in range(100):
    #     duc_docs.extend(duc_dataset['test'][i]['document'])
    # sample_document = duc_dataset['test'][0]['document']
    newsgroups = fetch_20newsgroups(subset='all', categories=['sci.space'], remove=('headers', 'footers', 'quotes'))
    docs = newsgroups.data
    text_sum = TextSummariser(docs)
    sents = text_sum.clean_text()
    summary = text_sum.generate_summary(100, sents)
    print(summary)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score('The quick brown fox jumps over the lazy dog',
                          'The quick brown dog jumps on the log.')
