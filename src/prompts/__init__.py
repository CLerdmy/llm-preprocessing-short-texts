from src.paths import PROMPTS


def load_prompt(name: str) -> str:
    path = PROMPTS / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"нет промпта {name}")
    return path.read_text(encoding="utf-8").strip()


def list_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPTS.glob("*.txt"))