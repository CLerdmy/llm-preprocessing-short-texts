"""Метрики качества тем"""

from itertools import combinations

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from src.paths import DATA

TOKENS = DATA / "topics" / "tokens"


class Reference:
    """Референс корпуса: тексты для подсчета совместной встречаемости слов"""

    def __init__(self, docs):
        self.docs = docs
        self.dictionary = Dictionary(docs)
        self.words = set(self.dictionary.token2id)

    @classmethod
    def load(cls, corpus):
        tokens = pd.read_parquet(TOKENS / f"reference_{corpus}.parquet").tokens
        return cls([t.split() for t in tokens])

    def coherence(self, topics, measure="c_npmi"):
        """Когерентность каждой темы"""
        known = [[w for w in topic if w in self.words] for topic in topics]
        usable = [i for i, topic in enumerate(known) if len(topic) >= 2]
        values = np.full(len(topics), np.nan)
        if usable:
            model = CoherenceModel(topics=[known[i] for i in usable], texts=self.docs,
                                   dictionary=self.dictionary, coherence=measure)
            values[usable] = model.get_coherence_per_topic()
        return values

    def out_of_reference(self, topics):
        """Доля слов тем, которых нет в референсе"""
        words = [w for topic in topics for w in topic]
        return sum(w not in self.words for w in words) / len(words)

    def random_topics(self, count=60, n=10, seed=42):
        """Случайные темы: слова выбираются с вероятностью, пропорциональной частоте в референсе"""
        rng = np.random.default_rng(seed)
        ids = np.array(list(self.dictionary.cfs))
        freq = np.array([self.dictionary.cfs[i] for i in ids], float)
        return [[self.dictionary[i] for i in rng.choice(ids, n, replace=False, p=freq / freq.sum())]
                for _ in range(count)]


def diversity(topics):
    """TD: доля уникальных слов среди топ-слов всех тем"""
    words = [w for topic in topics for w in topic]
    return len(set(words)) / len(words)


def uniqueness(topics):
    """TU: средняя по темам величина 1 / (число тем, где встречается слово)"""
    count = pd.Series([w for topic in topics for w in set(topic)]).value_counts()
    return float(np.mean([np.mean([1 / count[w] for w in topic]) for topic in topics]))


def redundancy(topics):
    """TR: среднее пересечение топ-слов по всем парам тем"""
    n = len(topics[0])
    return float(np.mean([len(set(a) & set(b)) / n for a, b in combinations(topics, 2)]))


def evaluate(result, reference, n=10):
    words = result.words(n)
    npmi = reference.coherence(words)
    return {"npmi": float(np.nanmean(npmi)),
            "вне референса": reference.out_of_reference(words),
            "TD": diversity(result.words(25)),
            "TU": uniqueness(words),
            "TR": redundancy(words)}


def stability(results, n=10):
    """Устойчивость между запусками с разным seed"""
    overlap, ari = [], []
    for a, b in combinations(results, 2):
        wa, wb = a.words(n), b.words(n)
        common = np.array([[len(set(x) & set(y)) / n for y in wb] for x in wa])
        rows, cols = linear_sum_assignment(-common)
        overlap.append(common[rows, cols].mean())
        ari.append(adjusted_rand_score(a.doc_topic.argmax(1), b.doc_topic.argmax(1)))
    return {"пересечение тем": float(np.mean(overlap)), "ARI": float(np.mean(ari))}