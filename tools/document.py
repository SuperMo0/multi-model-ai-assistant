from pathlib import Path
from pydantic import BaseModel
from pypdf import PdfReader

SUPPORTED = {".txt", ".pdf"}


class DocumentSummary(BaseModel):
    title: str
    main_topics: list[str]
    key_points: list[str]
    word_count: int
    sentiment: str
    recommended_actions: list[str]


def read_file(path: str) -> tuple[str, int]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{p.suffix}'. Supported: {sorted(SUPPORTED)}")

    if p.suffix == ".pdf":
        reader = PdfReader(str(p))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = p.read_text(encoding="utf-8")

    word_count = len(text.split())
    return text, word_count
