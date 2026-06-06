from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .text_utils import ascii_fold_turkish, normalize_for_eval


@dataclass
class EngineResult:
    engine: str
    text: str
    status: str
    error: str = ""


class BaseEngine:
    name = "base"
    needs_images = True

    def available(self) -> tuple[bool, str]:
        return True, ""

    def recognize(self, image_paths: Iterable[Path], reference_text: str = "") -> EngineResult:
        raise NotImplementedError


class MockNoisyEngine(BaseEngine):
    name = "mock"
    needs_images = False

    def recognize(self, image_paths: Iterable[Path], reference_text: str = "") -> EngineResult:
        noisy = ascii_fold_turkish(reference_text)
        noisy = noisy.replace("ölçüm", "olcum").replace("Ölçüm", "Olcum")
        return EngineResult(self.name, normalize_for_eval(noisy), "ok")


def _find_tesseract_executable() -> str | None:
    executable = shutil.which("tesseract")
    if executable:
        return executable
    common_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for path in common_paths:
        if path.exists():
            return str(path)
    return None


def _is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _find_tessdata_dir(language: str) -> Path | None:
    language_file = f"{language}.traineddata"
    project_tessdata = Path(__file__).resolve().parents[1] / "tools" / "tessdata"
    ascii_cache = Path(r"C:\ocrturk_tessdata")
    candidates: list[Path] = []
    if (project_tessdata / language_file).exists():
        if _is_ascii_path(project_tessdata):
            candidates.append(project_tessdata)
        else:
            try:
                ascii_cache.mkdir(parents=True, exist_ok=True)
                shutil.copy2(project_tessdata / language_file, ascii_cache / language_file)
                candidates.append(ascii_cache)
            except OSError:
                candidates.append(project_tessdata)
    candidates.append(ascii_cache)
    for tessdata_dir in candidates:
        if (tessdata_dir / language_file).exists():
            return tessdata_dir
    return None


class TesseractEngine(BaseEngine):
    name = "tesseract"

    def __init__(self, language: str = "tur", psm: int = 3):
        self.language = language
        self.psm = psm
        self.executable = _find_tesseract_executable()
        self.tessdata_dir = _find_tessdata_dir(language)

    def _tessdata_args(self) -> list[str]:
        if self.tessdata_dir is None:
            return []
        return ["--tessdata-dir", str(self.tessdata_dir)]

    def available(self) -> tuple[bool, str]:
        if self.executable is None:
            return False, "tesseract executable not found in PATH or common Windows install paths"
        try:
            proc = subprocess.run(
                [self.executable, *self._tessdata_args(), "--list-langs"],
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if proc.returncode != 0:
                return False, proc.stdout.strip()
            langs = set(proc.stdout.split())
            if self.language not in langs:
                return False, f"tesseract language `{self.language}` not installed"
        except Exception as exc:
            return False, str(exc)
        return True, ""

    def recognize(self, image_paths: Iterable[Path], reference_text: str = "") -> EngineResult:
        texts: list[str] = []
        for image_path in image_paths:
            proc = subprocess.run(
                [
                    self.executable or "tesseract",
                    *self._tessdata_args(),
                    str(image_path),
                    "stdout",
                    "-l",
                    self.language,
                    "--psm",
                    str(self.psm),
                ],
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                return EngineResult(self.name, "", "error", proc.stderr.strip() or proc.stdout.strip())
            texts.append(proc.stdout)
        return EngineResult(self.name, normalize_for_eval("\n".join(texts)), "ok")


class EasyOCREngine(BaseEngine):
    name = "easyocr"

    def __init__(self, gpu: bool = False):
        self.gpu = gpu
        self._reader = None

    def available(self) -> tuple[bool, str]:
        try:
            import easyocr  # noqa: F401
        except Exception as exc:
            return False, f"easyocr import failed: {exc}"
        return True, ""

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._reader = easyocr.Reader(["tr"], gpu=self.gpu)
        return self._reader

    def recognize(self, image_paths: Iterable[Path], reference_text: str = "") -> EngineResult:
        reader = self._get_reader()
        texts: list[str] = []
        for image_path in image_paths:
            import cv2
            import numpy as np

            image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if image is None:
                return EngineResult(self.name, "", "error", f"could not read image: {image_path}")
            result = reader.readtext(image, detail=0, paragraph=True)
            texts.extend(str(part) for part in result)
        return EngineResult(self.name, normalize_for_eval("\n".join(texts)), "ok")


class PaddleOCREngine(BaseEngine):
    name = "paddleocr"

    def available(self) -> tuple[bool, str]:
        try:
            import paddleocr  # noqa: F401
        except Exception as exc:
            return False, f"paddleocr import failed: {exc}"
        return True, ""

    def recognize(self, image_paths: Iterable[Path], reference_text: str = "") -> EngineResult:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(use_angle_cls=False, lang="tr", show_log=False)
        texts: list[str] = []
        for image_path in image_paths:
            result = ocr.ocr(str(image_path), cls=False)
            for page in result or []:
                for line in page or []:
                    if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                        texts.append(str(line[1][0]))
        return EngineResult(self.name, normalize_for_eval("\n".join(texts)), "ok")


def build_engine(name: str, gpu: bool = False) -> BaseEngine:
    normalized = name.lower()
    if normalized == "mock":
        return MockNoisyEngine()
    if normalized == "tesseract":
        return TesseractEngine()
    if normalized == "easyocr":
        return EasyOCREngine(gpu=gpu)
    if normalized == "paddleocr":
        return PaddleOCREngine()
    raise ValueError(f"Unknown OCR engine: {name}")
