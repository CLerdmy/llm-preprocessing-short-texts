"""Предобработка текстов для тематических моделей"""

import re
from collections import Counter
from functools import lru_cache

import nltk
import pandas as pd
import pymorphy3
from spacy.lang.ru.stop_words import STOP_WORDS as SPACY_RU

from src.paths import DATA

FINAL = DATA / "generation" / "final"
REFERENCE = DATA / "generation" / "reference.parquet"
OUT = DATA / "topics" / "tokens"

CORPORA = ["yt", "tg", "rr"]
VERSIONS = ["orig", "level1", "level2", "level3"]
MIN_LEN = 2                                             # минимальная длина леммы
MIN_LEMMAS = 2                                          # документ с меньшим числом лемм в любой версии исключается из всех версий
MIN_DF = 5   
MAX_DF = 0.4

WORD = re.compile(r"[А-Яа-яЁё]+|[A-Za-z]+")
CYRILLIC = re.compile(r"[а-я]")

MARKUP = re.compile("|".join([
    r"[\w.+-]+@[\w-]+\.[A-Za-z]+",                                              # почта
    r"https?://\S*",                                                            # ссылки
    r"\bwww\.\S+",
    r"\b[\w.-]+\.(?:com|ru|org|net|io|dev|ai|me|su|info|tv|gg|app|рф)\b\S*",    # домены с путем
    r"@\w+",                                                                    # упоминания и заглушки @user_...
    r"(?:~|\.)?(?:/[A-Za-z_][\w.-]*){2,}/?",                                    # пути unix
    r"\b[A-Za-z]:\\\S+"                                                         # пути windows
]))


def strip_markup(text):
    return MARKUP.sub(" ", str(text))

nltk.download("stopwords", quiet=True)
morph = pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=None)
def lemma(word):
    """Кириллица - нормальная форма pymorphy3"""
    if CYRILLIC.match(word):
        return morph.parse(word)[0].normal_form.replace("ё", "е")
    return word


STOP_FORMS = {w.replace("ё", "е") for w in set(nltk.corpus.stopwords.words("russian")) | set(nltk.corpus.stopwords.words("english")) | set(SPACY_RU)}
STOP = STOP_FORMS


def lemmatize(text):
    """Текст -> список лемм"""
    result = []
    for word in WORD.findall(strip_markup(text).lower().replace("ё", "е")):
        if word in STOP:
            continue
        lem = lemma(word)
        if lem not in STOP and len(lem) >= MIN_LEN:
            result.append(lem)
    return result


def vocabulary(docs):
    """Словарь с порогами частоты"""
    df = Counter(w for doc in docs for w in set(doc))
    return {w for w, n in df.items() if n >= MIN_DF and n / len(docs) <= MAX_DF}


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    reference_all = pd.read_parquet(REFERENCE)

    for corpus in CORPORA:
        df = pd.read_parquet(FINAL / f"{corpus}.parquet")
        tokens = {v: df[v].map(lemmatize) for v in VERSIONS}

        keep = pd.concat([tokens[v].map(len) >= MIN_LEMMAS for v in VERSIONS], axis=1).all(axis=1)
        out = pd.DataFrame({"uid": df.uid[keep].values})
        for v in VERSIONS:
            out[v] = tokens[v][keep].map(" ".join).values
        out.to_parquet(OUT / f"{corpus}.parquet", index=False)

        ref = reference_all[reference_all.corpus == corpus]
        ref_tokens = ref.text.map(lemmatize)
        ref_out = pd.DataFrame({"uid": ref.uid.values, "tokens": ref_tokens.map(" ".join).values})
        ref_out = ref_out[ref_tokens.map(len).values >= MIN_LEMMAS]
        ref_out.to_parquet(OUT / f"reference_{corpus}.parquet", index=False)
        ref_words = {w for t in ref_out.tokens for w in t.split()}

        print(f"\n{corpus}: документов {len(df)}, исключено {int((~keep).sum())}, "
              f"осталось {len(out)}; референс {len(ref_out)} из {len(ref)}")
        rows = []
        for v in VERSIONS:
            docs = [t.split() for t in out[v]]
            words = [w for doc in docs for w in doc]
            vocab = vocabulary(docs)
            rows.append({"версия": v, "лемм": len(words), "уникальных": len(set(words)),
                         "словарь": len(vocab),
                         "латиница": f"{sum(not CYRILLIC.match(w) for w in words) / len(words):.1%}",
                         "нет в референсе": f"{len(vocab - ref_words) / max(len(vocab), 1):.1%}"})
        print(pd.DataFrame(rows).to_string(index=False))

        for v in VERSIONS:
            top = Counter(w for t in out[v] for w in t.split()).most_common(15)
            print(f"    {v}: {', '.join(w for w, _ in top)}")
        short = Counter(w for t in out.orig for w in t.split() if len(w) == 2).most_common(20)
        print(f"    двухбуквенные в orig: {', '.join(f'{w} {n}' for w, n in short)}")


if __name__ == "__main__":
    build()