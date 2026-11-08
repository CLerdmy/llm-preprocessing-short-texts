"""Генерация трех условий на рабочей выборке: первый запуск.

Документы идут по одному, три условия подряд.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from tqdm import tqdm

from src.llm import GigaChat
from src.paths import DATA

GEN = DATA / "generation"
LEVELS = ["level1", "level2", "level3"]
MODEL = "GigaChat-3-Lightning"
PAUSE = 30               
MAX_ERRORS_IN_ROW = 10

MAX_ATTEMPTS = 3
RETRY = {"error", "missing", "empty", "placeholder", "length"}


def append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_state(path):
    last, attempts = {}, {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r["uid"], r["level"])
                attempts[key] = attempts.get(key, 0) + 1
                if last.get(key) != "ok":
                    last[key] = r["status"]
    return last, attempts


def process(llm, text, prompt):
    llm.last = {}
    try:
        res, log = llm.ask_batch(pd.DataFrame({"id": [1], "text": [text]}), prompt, batch=1, temperature=0.0)
        error = log["error"].iloc[0]
        out = res.llm_text.iloc[0]
        out = None if pd.isna(out) else out
    except Exception as e:
        error, out = f"{type(e).__name__}: {e}"[:300], None
    info = llm.last
    reason = info.get("finish_reason")

    if error:
        status = "error"
    elif reason == "blacklist":
        status = "blocked"
    elif reason == "length":
        status = "length"
    elif not out or not str(out).strip():
        status = "missing"
    elif str(out).strip().startswith("[") and str(out).strip().endswith("]"):
        status = "placeholder"
    else:
        status = "ok"

    return status, {
        "text": out if status == "ok" else None, "error": error, "finish_reason": reason,
        "model_version": info.get("model_version"), "tokens": info.get("tokens", 0),
        "precached_tokens": info.get("precached_tokens", 0), "elapsed": info.get("elapsed"),
        "raw": info.get("text") if status != "ok" else None
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="main")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--blocked", action="store_true")
    parser.add_argument("--source", default="work.parquet")
    args = parser.parse_args()

    run_dir = GEN / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    results = run_dir / "results.jsonl"
    logging.basicConfig(filename=run_dir / "generation.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")

    manifest = json.loads((GEN / "manifest.json").read_text(encoding="utf-8"))
    prompts = {}
    for level in LEVELS:
        path = GEN / "prompts" / f"{level}.txt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["prompts_sha256"][level], \
            f"{path} изменен после фиксации"
        prompts[level] = path.read_text(encoding="utf-8")

    docs = pd.read_parquet(GEN / args.source).sort_values("order")
    if args.limit:
        docs = docs.head(args.limit)
    last, attempts = read_state(results)
    retry = RETRY | ({"blocked"} if args.blocked else set())

    def need(uid, level):
        key = (uid, level)
        if args.round == 1:
            return key not in last
        return last.get(key) in retry and attempts.get(key, 0) < MAX_ATTEMPTS

    todo = [d for d in docs.itertuples() if any(need(d.uid, lv) for lv in LEVELS)]
    logging.info(f"старт круга {args.round}: документов {len(docs)}, к отправке {len(todo)}")

    llm = GigaChat(model=MODEL)
    counts = {"ok": 0, "blocked": 0, "other": 0}
    errors_in_row = 0
    total = len(docs) if args.round == 1 else len(todo)
    bar = tqdm(total=total, initial=total - len(todo), unit="док", smoothing=0.05)

    try:
        for doc in todo:
            for level in LEVELS:
                if not need(doc.uid, level):
                    continue
                status, fields = process(llm, doc.text, prompts[level])
                append(results, {"ts": datetime.now().isoformat(timespec="seconds"), "uid": doc.uid,
                                 "level": level, "status": status, "round": args.round, **fields})
                counts[status if status in counts else "other"] += 1

                if status == "error":
                    errors_in_row += 1
                    logging.info(f"{doc.uid} {level}: {fields['error']}")
                    if errors_in_row >= MAX_ERRORS_IN_ROW:
                        sys.exit(f"\n{errors_in_row} ошибок подряд, остановка")
                    time.sleep(PAUSE)
                else:
                    errors_in_row = 0
                    if status != "ok":
                        logging.info(f"{doc.uid} {level}: {status}")
            bar.update(1)
            bar.set_postfix(counts)
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        bar.close()
        logging.info(f"конец сессии: {counts}")


if __name__ == "__main__":
    main()