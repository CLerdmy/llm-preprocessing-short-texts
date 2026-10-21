"""Пути проекта"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

PROMPTS = ROOT / "src" / "prompts"
ENV = ROOT / ".env"