"""Задания для разметки потерь содержания

Создает Excel-файлы для разметчиков. Карточки общего ядра у всех одинаковые,
индивидуальные не повторяются. Все выданные карточки записываются в registry.csv, поэтому
повторный запуск продолжает нумерацию и не выдает уже использованные документы.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from src.paths import DATA

FINAL = DATA / "generation" / "final"
OUT = DATA / "annotation" / "losses"
REGISTRY = OUT / "registry.csv"

CORPORA = ["yt", "tg", "rr"]
LEVELS = ["level1", "level2", "level3"]
LENGTHS = ["короткий", "средний", "длинный"]
SEED = 42
CORE_PER_CELL = 4   # общее ядро: карточек на корпус и условие, всего 36
OWN_PER_CELL = 2    # индивидуальные: карточек на корпус и условие, 18 на разметчика

YES_NO_NA = ["да", "нет", "не применимо"]
QUESTIONS = [
    ("1. Сохранены ли участники и объекты?", YES_NO_NA),
    ("2. Сохранены ли числа и количественные отношения?", YES_NO_NA),
    ("3. Сохранены ли отрицания и полярность утверждений?", YES_NO_NA),
    ("4. Сохранены ли позиция и намерение автора?", YES_NO_NA),
    ("5. Обошлось ли без нового содержания?", ["да", "нет"]),
    ("6. Сохранено ли содержание в целом?",
     ["полностью сохранено", "в основном сохранено", "частично сохранено", "не сохранено"]),
    ("Не могу оценить", ["не могу оценить"]),
]

WORD = re.compile(r"[А-Яа-яЁё]+")


def norm(text):
    return " ".join(str(text).split())


def load_corpus():
    """Итоговые корпуса с группой длины: трети длины исходного текста внутри корпуса"""
    df = pd.concat([pd.read_parquet(FINAL / f"{c}.parquet").assign(corpus=c) for c in CORPORA],
                   ignore_index=True)
    df["n_words"] = df.orig.map(lambda t: len(WORD.findall(str(t))))
    df["length"] = df.groupby("corpus").n_words.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=LENGTHS).astype(str))
    return df


def draw(df, used, per_cell, rng, shift):
    """per_cell документов на каждое сочетание корпуса и условия"""
    rows = []
    for cell, (corpus, level) in enumerate((c, lv) for c in CORPORA for lv in LEVELS):
        changed = (df.corpus == corpus) & (df.orig.map(norm) != df[level].map(norm))
        for i in range(per_cell):
            length = LENGTHS[(cell + shift + i) % len(LENGTHS)]
            pool = df[changed & (df.length == length) & ~df.uid.isin(used)]
            doc = pool.iloc[rng.integers(len(pool))]
            used.add(doc.uid)
            rows.append({"uid": doc.uid, "corpus": corpus, "level": level, "length": length,
                         "orig": norm(doc.orig), "result": norm(doc[level])})
    return rows


def add(registry, rows, part, annotator):
    new = pd.DataFrame(rows)
    new.insert(0, "id", [f"{i:03d}" for i in range(len(registry) + 1, len(registry) + len(new) + 1)])
    new["part"] = part
    new["annotator"] = annotator
    return pd.concat([registry, new], ignore_index=True)


def write_excel(cards, path):
    wb = Workbook()
    ws = wb.active
    ws.title = "Разметка"
    ws.append(["id", "Исходный текст", "Преобразованный текст"] + [q for q, _ in QUESTIONS])
    for card in cards.itertuples():
        ws.append([card.id, card.orig, card.result] + [None] * len(QUESTIONS))

    font, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    wrap = Alignment(wrap_text=True, vertical="top")
    answer_fill = PatternFill("solid", fgColor="FFF7E0")
    header_fill = PatternFill("solid", fgColor="E8EDF2")
    last = ws.max_row

    for col, width in enumerate([7, 45, 45] + [17] * len(QUESTIONS), start=1):
        ws.column_dimensions[get_column_letter(col)].width = width
    for row in ws.iter_rows():
        for cell in row:
            cell.font, cell.alignment = font, wrap
    for cell in ws[1]:
        cell.font, cell.fill = bold, header_fill
    ws.row_dimensions[1].height = 54
    for r in range(2, last + 1):
        longest = max(len(str(ws.cell(r, 2).value)), len(str(ws.cell(r, 3).value)))
        ws.row_dimensions[r].height = 14 * (longest // 50 + 1) + 4

    for col, (title, options) in enumerate(QUESTIONS, start=4):
        letter = get_column_letter(col)
        rule = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True,
                              showErrorMessage=True, errorTitle="Недопустимое значение",
                              error="Выберите значение из списка.")
        ws.add_data_validation(rule)
        rule.add(f"{letter}2:{letter}{last}")
        for r in range(2, last + 1):
            ws.cell(r, col).protection = Protection(locked=False)
            ws.cell(r, col).fill = answer_fill

    ws.freeze_panes = "B2"
    ws.protection.sheet = True
    ws.protection.formatColumns = False
    ws.protection.formatRows = False
    wb.save(path)


def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        sys.exit("укажите число разметчиков: python scripts/make_loss_tasks.py 3")
    count = int(sys.argv[1])

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_corpus()
    registry = pd.read_csv(REGISTRY, dtype={"id": str}) if REGISTRY.exists() else pd.DataFrame()
    used = set(registry.uid) if len(registry) else set()

    if registry.empty:
        core = draw(df, used, CORE_PER_CELL, np.random.default_rng(SEED), shift=0)
        registry = add(registry, core, "core", 0)
        registry.to_csv(REGISTRY, index=False)
        print(f"общее ядро: {len(core)} карточек")

    first = int(registry.annotator.max()) + 1
    for annotator in range(first, first + count):
        own = draw(df, used, OWN_PER_CELL, np.random.default_rng(SEED + annotator), shift=annotator)
        registry = add(registry, own, "own", annotator)
        registry.to_csv(REGISTRY, index=False)

        cards = registry[registry.annotator.isin([0, annotator])].sample(frac=1, random_state=SEED + annotator)
        path = OUT / f"losses_{annotator:02d}.xlsx"
        write_excel(cards, path)
        print(f"разметчик {annotator:02d}: {len(cards)} карточек -> {path}")

    print(f"всего карточек в {REGISTRY}: {len(registry)}")


if __name__ == "__main__":
    main()
