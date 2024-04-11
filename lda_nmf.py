from typing import List

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from datasets import load_dataset


class TopicModels:
    """
    This class runs LDA and NMF library implementations on 20NG dataset, returns top 20 words in dataset with their
    respective probabilities.
    """

    def __init__(self, doc: List[str]) -> None:
        """
        Initialise an object of CountVectorizer - this converts a collection of text documents to a matrix of token
        counts.
        """
        self.vectors = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.word_vecs = self.vectors.fit_transform(doc)
        self.feature_names = self.vectors.get_feature_names_out()
        self.model = None

    def latent_dirichlet_library(self, components: int) -> None:
        """
        Run LDA library implementation on cleaned 20NG dataset.
        :param components: no.of components for LDA
        """
        lda = LatentDirichletAllocation(n_components=components, random_state=42)
        lda.fit(self.word_vecs)
        # set the model to nmf
        self.model = lda

    def non_negative_matrix_factorization(self, components: int) -> None:
        """
        Run NMF library implementation on cleaned 20NG dataset.
        :param components: no.of components for LDA
        """
        nmf = NMF(n_components=components, random_state=42)
        nmf.fit(self.word_vecs)
        # set the model to nmf
        self.model = nmf

    def display_top_20_words(self, n_top_words: int = 20) -> None:
        """
        Print top 20 words in each topic.
        :param n_top_words: number of words
        """
        print(f"model - "
              f"{'lda' if isinstance(self.model, LatentDirichletAllocation) else 'nmf'}")
        for topic_idx, topic in enumerate(self.model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([self.feature_names[i] + "-" +
                                 str(topic[i]) for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        return


if __name__ == "__main__":
    duc_dataset = load_dataset("midas/duc2001")
    duc_docs = []
    for i in range(100):
        duc_docs.extend(duc_dataset['test'][i]['document'])
    sample_document = duc_dataset['test'][0]['document']

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                          remove=('headers', 'footers', 'quotes'))
    # topic_models = TopicModels(duc_docs)
    print(len(newsgroups_train.data[0]))
    topic_models = TopicModels(newsgroups_train.data)
    for comp in [10, 20, 50]:
        print(f"Number of components - {comp}")
        # run and display result of LDA
        topic_models.latent_dirichlet_library(comp)
        topic_models.display_top_20_words()
        # run and display result of NMF
        topic_models.non_negative_matrix_factorization(comp)
        topic_models.display_top_20_words()


