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
    topics: list                    # K тем, в каждой TOP_N пар (слово, вес) по убыванию веса
    doc_topic: np.ndarray           # документы / темы
    params: dict
    seconds: float
    topic_word: np.ndarray = None   # темы / словарь: полное распределение слов (у BERTopic — веса c-TF-IDF)
    vocab: list = None              # словарь модели в порядке столбцов topic_word

    def words(self, n=10):
        """Топ-n слов каждой темы"""
        return [[w for w, _ in topic[:n]] for topic in self.topics]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        info = {"topics": self.topics, "params": self.params, "seconds": self.seconds}
        (path / "topics.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        np.save(path / "doc_topic.npy", self.doc_topic.astype(np.float32))
        if self.topic_word is not None:
            np.save(path / "topic_word.npy", self.topic_word.astype(np.float32))
            (path / "vocab.json").write_text(json.dumps(list(self.vocab), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        info = json.loads((path / "topics.json").read_text(encoding="utf-8"))
        topics = [[(w, v) for w, v in topic] for topic in info["topics"]]
        topic_word = vocab = None
        if (path / "topic_word.npy").exists():
            topic_word = np.load(path / "topic_word.npy")
            vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        return cls(topics, np.load(path / "doc_topic.npy"), info["params"], info["seconds"], topic_word, vocab)


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
        self.model = None

    def fit(self, docs, texts=None, embeddings=None):
        """_fit возвращает (темы, документ / тема) или (темы, документ / тема, тема / слово, словарь)"""
        start = time.time()
        out = self._fit(docs, texts, embeddings)
        topics, doc_topic = out[:2]
        topic_word, vocab = out[2:] if len(out) == 4 else (None, None)
        params = {"model": self.name, "k": self.k, "seed": self.seed, **self.params}
        return TopicResult(topics, np.asarray(doc_topic), params, round(time.time() - start, 1),
                           None if topic_word is None else np.asarray(topic_word, dtype=np.float32),
                           None if vocab is None else [str(w) for w in vocab])

    def _fit(self, docs, texts, embeddings):
        raise NotImplementedError