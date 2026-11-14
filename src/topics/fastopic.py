"""FASTopic"""

import random

import numpy as np
import torch
from fastopic import FASTopic as _FASTopic
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from src.topics.base import TOP_N, TopicModel, build_vocabulary

K = 20
SEED = 42
SCALE = 10
EPOCHS = 800
LEARNING_RATE = 0.002
DT_ALPHA = 3.0
TW_ALPHA = 2.0
THETA_TEMP = 1.0


class _Vectors:
    """Готовые векторы вместо модели векторов FASTopic"""

    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, docs, *args, **kwargs):
        assert len(docs) == len(self.vectors), "число документов и векторов не совпадает"
        return self.vectors


class _Bow:
    """Готовый мешок слов вместо предобработки FASTopic"""

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def preprocess(self, docs, *args, **kwargs):
        vectorizer = CountVectorizer(vocabulary=self.vocabulary, token_pattern=r"\S+", lowercase=False)
        bow = sparse.csr_matrix(vectorizer.transform(docs))
        return {"train_bow": bow, "vocab": list(self.vocabulary), "train_texts": docs}


class FASTopic(TopicModel):
    name = "fastopic"

    def __init__(self, k=K, seed=SEED, scale=SCALE, epochs=EPOCHS, learning_rate=LEARNING_RATE,
                 dt_alpha=DT_ALPHA, tw_alpha=TW_ALPHA, theta_temp=THETA_TEMP):
        super().__init__(k, seed, scale=scale, epochs=epochs, learning_rate=learning_rate,
                         dt_alpha=dt_alpha, tw_alpha=tw_alpha, theta_temp=theta_temp)

    def _fit(self, docs, texts, embeddings):
        if embeddings is None:
            raise ValueError("FASTopic нужны векторы документов: load_embeddings(corpus, version, uids)")
        p = self.params

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        lines = [" ".join(doc) for doc in docs]
        vocabulary = sorted(build_vocabulary(docs))
        vectors = embeddings * np.float32(p["scale"])
        model = _FASTopic(self.k, preprocess=_Bow(vocabulary), doc_embed_model=_Vectors(vectors),
                         num_top_words=TOP_N, device="cuda" if torch.cuda.is_available() else "cpu",
                         verbose=False, DT_alpha=p["dt_alpha"], TW_alpha=p["tw_alpha"], theta_temp=p["theta_temp"])
        _, theta = model.fit_transform(lines, epochs=p["epochs"], learning_rate=p["learning_rate"])

        beta = np.asarray(model.get_beta())
        vocab = model.vocab
        topics = [[(str(vocab[j]), float(beta[t, j])) for j in np.argsort(-beta[t])[:TOP_N]]
                  for t in range(self.k)]
        return topics, np.asarray(theta, dtype=float)