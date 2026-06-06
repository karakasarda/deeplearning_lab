from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import OCRTURK_REPO_URL
from .text_utils import strip_markdown_to_text


@dataclass(frozen=True)
class DataItem:
    item_id: str
    root: Path
    markdown_path: Path
    pdf_path: Path | None
    source_path: Path | None
    count_info_path: Path | None
    document_type: str
    difficulty: str
    source: str


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ensure_ocrturk_dataset(dataset_root: Path) -> None:
    if (dataset_root / ".git").exists() or (dataset_root / "data").exists():
        return
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", OCRTURK_REPO_URL, str(dataset_root)],
        check=True,
    )


def _candidate_data_root(dataset_root: Path) -> Path:
    if (dataset_root / "data").exists():
        return dataset_root / "data"
    return dataset_root


def discover_items(dataset_root: Path) -> list[DataItem]:
    data_root = _candidate_data_root(dataset_root)
    if not data_root.exists():
        return []
    items: list[DataItem] = []
    for item_root in sorted(data_root.glob("data_*"), key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else p.name):
        if not item_root.is_dir():
            continue
        item_id = item_root.name
        markdown_candidates = sorted(item_root.glob("*.md"))
        if not markdown_candidates:
            continue
        pdf_candidates = sorted(item_root.glob("*.pdf"))
        source_path = item_root / "source.json"
        count_info_path = item_root / "count_info.json"
        source = read_json(source_path)
        items.append(
            DataItem(
                item_id=item_id,
                root=item_root,
                markdown_path=markdown_candidates[0],
                pdf_path=pdf_candidates[0] if pdf_candidates else None,
                source_path=source_path if source_path.exists() else None,
                count_info_path=count_info_path if count_info_path.exists() else None,
                document_type=str(source.get("type", "unknown")),
                difficulty=str(source.get("difficulty", "unknown")),
                source=str(source.get("source", "")),
            )
        )
    return items


def load_reference_text(item: DataItem) -> str:
    return strip_markdown_to_text(item.markdown_path.read_text(encoding="utf-8-sig"))


def render_pdf_pages(item: DataItem, output_root: Path, dpi: int, force: bool = False) -> list[Path]:
    if item.pdf_path is None:
        raise FileNotFoundError(f"No PDF file found for {item.item_id}")
    try:
        import fitz  # PyMuPDF
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("PyMuPDF is required for rendering. Install with `pip install pymupdf`.") from exc

    item_output = output_root / item.item_id
    item_output.mkdir(parents=True, exist_ok=True)
    with fitz.open(item.pdf_path) as document:
        paths: list[Path] = []
        for page_index, page in enumerate(document):
            out_path = item_output / f"{item.item_id}_page_{page_index + 1}.png"
            if out_path.exists() and not force:
                paths.append(out_path)
                continue
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            pix.save(out_path)
            paths.append(out_path)
    return paths


def create_demo_dataset(dataset_root: Path) -> None:
    data_dir = dataset_root / "data" / "data_1"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "data_1.md").write_text(
        (
            "Türkçe OCR ölçümü için küçük bir deneme sayfası.\n"
            "Türkiye ekonomisi, üniversite raporları ve ölçüm sonuçları diakritik hata analizi için kullanılır.\n"
            "Bu örnek gerçek OCRTurk verisi değildir; yalnızca smoke test içindir.\n"
        ),
        encoding="utf-8",
    )
    (data_dir / "source.json").write_text(
        json.dumps({"difficulty": "demo", "type": "demo", "source": "synthetic"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (data_dir / "count_info.json").write_text(
        json.dumps({"equations": 0, "figures": 0, "tables": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
