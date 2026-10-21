"""Интерфейс LLM"""

from abc import ABC, abstractmethod
import re

import pandas as pd
from tqdm.auto import tqdm

from src.prompts import load_prompt


LINE = re.compile(r"\s*(\d+)\s*(?:\|\|\||\t|[—–\-:.)])\s*(.+)")
SPACES = re.compile(r"\s+")


def one_line(text) -> str:
    return SPACES.sub(" ", str(text)).strip()


class BaseLLM(ABC):

    @abstractmethod
    def ask(self, text: str, prompt: str | None = None, temperature: float | None = None) -> dict:
        """Запрос к LLM"""

    def ask_batch(self, df: pd.DataFrame, prompt: str, batch: int = 30, temperature: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Пакетная обработка"""

        full_prompt = prompt + "\n\n" + load_prompt("batch_rule")
        chunks = [df.iloc[i:i + batch] for i in range(0, len(df), batch)]
        answers, log = {}, []

        for n, chunk in enumerate(tqdm(chunks)):
            body = "\n".join(f"{row.id} ||| {one_line(row.text)}" for row in chunk.itertuples())
            try:
                answer = self.ask(body, prompt=full_prompt, temperature=temperature)
            except Exception as error:
                log.append({"batch": n, "sent": len(chunk), "got": 0, "tokens": 0, "elapsed": 0.0, "error": repr(error)[:80]})
                continue

            got = {}
            for line in answer["text"].splitlines():
                match = LINE.match(line)
                if match:
                    got[match.group(1)] = match.group(2).strip()

            answers.update(got)
            log.append({"batch": n, "sent": len(chunk), "got": len(got), "tokens": answer["tokens"], "elapsed": answer["elapsed"], "error": None})

        result = df.copy()
        result["llm_text"] = [answers.get(str(i)) for i in result["id"]]
        return result, pd.DataFrame(log)