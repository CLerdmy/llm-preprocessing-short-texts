"""Общий интерфейс тематических моделей

Модель получает документы в виде лемм и возвращает результат одного формата: 
темы со словами и весами и матрицу документ / тема
"""

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_DF = 5
MAX_DF = 0.4
TOP_N = 25


@dataclass
class TopicResult:
    topics: list                # K тем, в каждой TOP_N пар (слово, вес) по убыванию веса
    doc_topic: np.ndarray       # документы / темы
    params: dict
    seconds: float

    def words(self, n=10):
        """Топ-n слов каждой темы"""
        return [[w for w, _ in topic[:n]] for topic in self.topics]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        info = {"topics": self.topics, "params": self.params, "seconds": self.seconds}
        (path / "topics.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        np.save(path / "doc_topic.npy", self.doc_topic.astype(np.float32))

    @classmethod
    def load(cls, path):
        path = Path(path)
        info = json.loads((path / "topics.json").read_text(encoding="utf-8"))
        topics = [[(w, v) for w, v in topic] for topic in info["topics"]]
        return cls(topics, np.load(path / "doc_topic.npy"), info["params"], info["seconds"])


def filter_vocabulary(docs, min_df=MIN_DF, max_df=MAX_DF):
    df = Counter(w for doc in docs for w in set(doc))
    vocab = {w for w, n in df.items() if n >= min_df and n / len(docs) <= max_df}
    return [[w for w in doc if w in vocab] for doc in docs]


def build_vocabulary(docs, min_df=MIN_DF, max_df=MAX_DF):
    """Словарь с порогами частоты"""
    df = Counter(w for doc in docs for w in set(doc))
    return {w for w, n in df.items() if n >= min_df and n / len(docs) <= max_df}


def load_embeddings(corpus, version, uids):
    """Векторы документов версии корпуса в порядке uids"""
    import pandas as pd
    from src.paths import DATA
    table = pd.read_parquet(DATA / "embeddings" / f"{corpus}_{version}.parquet").set_index("uid")
    return table.loc[list(uids)].to_numpy(dtype=np.float32)


class TopicModel:

    name = "base"

    def __init__(self, k=20, seed=42, **params):
        self.k = k
        self.seed = seed
        self.params = params

    def fit(self, docs, texts=None, embeddings=None):
        start = time.time()
        topics, doc_topic = self._fit(docs, texts, embeddings)
        params = {"model": self.name, "k": self.k, "seed": self.seed, **self.params}
        return TopicResult(topics, doc_topic, params, round(time.time() - start, 1))

    def _fit(self, docs, texts, embeddings):
        raise NotImplementedError