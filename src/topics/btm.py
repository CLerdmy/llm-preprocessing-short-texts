"""BTM (Biterm Topic Model)"""

import bitermplus as btm
import numpy as np

from src.topics.base import MAX_DF, MIN_DF, TOP_N, TopicModel

K = 20
SEED = 42
ITERATIONS = 600
ALPHA = None
BETA = 0.01
WINDOW = 15


class BTM(TopicModel):
    name = "btm"

    def __init__(self, k=K, seed=SEED, iterations=ITERATIONS, alpha=ALPHA, beta=BETA, window=WINDOW):
        alpha = 50 / k if alpha is None else alpha
        super().__init__(k, seed, iterations=iterations, alpha=alpha, beta=beta, window=window)

    def _fit(self, docs, texts, embeddings):
        p = self.params
        lines = [" ".join(doc) for doc in docs]
        counts, vocabulary, _ = btm.get_words_freqs(lines, min_df=MIN_DF, max_df=MAX_DF, token_pattern=r"\S+")
        vectorized = btm.get_vectorized_docs(lines, vocabulary)
        biterms = btm.get_biterms(vectorized, p["window"])

        model = btm.BTM(counts, vocabulary, T=self.k, M=10, alpha=p["alpha"], beta=p["beta"],
                        seed=self.seed, win=p["window"])
        model.fit(biterms, iterations=p["iterations"], verbose=False)

        phi = model.matrix_topics_words_
        topics = [[(str(vocabulary[j]), float(phi[t, j])) for j in np.argsort(-phi[t])[:TOP_N]]
                  for t in range(self.k)]
        doc_topic = np.nan_to_num(model.transform(vectorized, verbose=False), nan=1 / self.k)
        self.model = model
        return topics, doc_topic, phi, list(vocabulary)