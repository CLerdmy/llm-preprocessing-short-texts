"""BERTopic"""

import numpy as np
from bertopic import BERTopic as _BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.topics.base import TOP_N, TopicModel, build_vocabulary

K = 20
SEED = 42
USE_UMAP = True
N_NEIGHBORS = 15
N_COMPONENTS = 5
REDUCE_FREQUENT_WORDS = False


class BERTopic(TopicModel):
    name = "bertopic"
 
    def __init__(self, k=K, seed=SEED, use_umap=USE_UMAP, n_neighbors=N_NEIGHBORS,
                 n_components=N_COMPONENTS, reduce_frequent_words=REDUCE_FREQUENT_WORDS):
        super().__init__(k, seed, use_umap=use_umap, n_neighbors=n_neighbors,
                         n_components=n_components, reduce_frequent_words=reduce_frequent_words)
 
    def _fit(self, docs, texts, embeddings):
        if embeddings is None:
            raise ValueError("BERTopic нужны векторы документов: load_embeddings(corpus, version, uids)")
        p = self.params
        lines = [" ".join(doc) for doc in docs]
 
        if p["use_umap"]:
            reducer = UMAP(n_neighbors=p["n_neighbors"], n_components=p["n_components"], min_dist=0.0,
                           metric="cosine", random_state=self.seed)
        else:
            reducer = BaseDimensionalityReduction()
 
        vocabulary = sorted(build_vocabulary(docs))
        model = _BERTopic(umap_model=reducer,
                         hdbscan_model=KMeans(n_clusters=self.k, random_state=self.seed, n_init=10),
                         vectorizer_model=CountVectorizer(vocabulary=vocabulary, token_pattern=r"\S+", lowercase=False),
                         ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=p["reduce_frequent_words"]),
                         top_n_words=TOP_N, calculate_probabilities=False,
                         language="multilingual")
        model.fit(lines, embeddings=embeddings)
 
        topics = [[(w, float(s)) for w, s in model.get_topic(t)][:TOP_N] for t in range(self.k)]
 
        topic_vectors = model.topic_embeddings_ / np.linalg.norm(model.topic_embeddings_, axis=1, keepdims=True)
        doc_topic = np.clip(embeddings @ topic_vectors.T, 0, None)
        doc_topic = doc_topic / doc_topic.sum(axis=1, keepdims=True)
 
        topic_word = model.c_tf_idf_[model._outliers:model._outliers + self.k].toarray()
        self.model = model
        return topics, doc_topic, topic_word, list(model.vectorizer_model.get_feature_names_out())