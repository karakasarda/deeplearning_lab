from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.config import OCRTURK_PAPER_URL, OCRTURK_REPO_URL, PROJECT_REPO_URL


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "final"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "output" / "pdf" / "Project5_OCRTurk_IEEE_Format_Report.pdf"


REFERENCES = [
    (
        "D. Yılmaz, E. A. Munis, C. Toraman, S. K. Köse, B. Aktaş, M. C. Baytekin, and B. K. Görür, \"OCRTurk: A Comprehensive OCR Benchmark for Turkish,\" in Proceedings of SIGTURK 2026, pp. 197-208, 2026, doi: 10.18653/v1/2026.sigturk-1.16.",
        OCRTURK_PAPER_URL,
    ),
    ("METU NLP, \"OCRTurk dataset repository,\" GitHub, 2026.", "https://github.com/metunlp/ocrturk"),
    (
        "M. G. Öztürk, D. Ö. Şahin, and E. Kılıç, \"Turkish Optical Character Recognition Under the Lens: A Systematic Review of Language-Specific Challenges, Dataset Scarcity, and Open-Source Limitations,\" IEEE Access, vol. 13, pp. 168977-168997, 2025, doi: 10.1109/ACCESS.2025.3614147.",
        "https://doi.org/10.1109/ACCESS.2025.3614147",
    ),
    (
        "Y. Yılmaz, E. G. Hanoğlu, A. G. Özkan, and K. Öztoprak, \"Benchmarking OCR and Vision-Language Models for Turkish Text Recognition: A Comprehensive Evaluation Using Synthetic Data,\" Research Square preprint, 2025, doi: 10.21203/rs.3.rs-7797886/v1.",
        "https://doi.org/10.21203/rs.3.rs-7797886/v1",
    ),
    ("Tesseract OCR contributors, \"Tesseract OCR documentation,\" 2026.", "https://tesseract-ocr.github.io/"),
    ("Jaided AI, \"EasyOCR: Ready-to-use OCR with 80+ supported languages,\" GitHub repository, 2024.", "https://github.com/JaidedAI/EasyOCR"),
    ("PaddlePaddle, \"PaddleOCR: multilingual OCR and document parsing toolkit,\" GitHub repository, 2026.", "https://github.com/PaddlePaddle/PaddleOCR"),
    ("MDZ Digital Library team, \"dbmdz/bert-base-turkish-cased: BERTurk model card,\" Hugging Face, 2026.", "https://huggingface.co/dbmdz/bert-base-turkish-cased"),
    (
        "L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel, \"mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer,\" in Proceedings of NAACL-HLT 2021, pp. 483-498, 2021, doi: 10.18653/v1/2021.naacl-main.41.",
        "https://aclanthology.org/2021.naacl-main.41/",
    ),
    (
        "C. Sayallar, A. Sayar, and N. Babalık, \"An OCR Engine for Printed Receipt Images using Deep Learning Techniques,\" International Journal of Advanced Computer Science and Applications, vol. 14, no. 2, pp. 833-840, 2023, doi: 10.14569/IJACSA.2023.0140295.",
        "https://doi.org/10.14569/IJACSA.2023.0140295",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Project5 in the sample IEEE-style paper PDF format")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--author", default="Arda Karakaş")
    parser.add_argument("--affiliation", default="Ankara Üniversitesi, YZM304 Derin Öğrenme")
    return parser.parse_args()


def register_fonts() -> dict[str, str]:
    font_paths = {
        "regular": Path(r"C:\Windows\Fonts\times.ttf"),
        "bold": Path(r"C:\Windows\Fonts\timesbd.ttf"),
        "italic": Path(r"C:\Windows\Fonts\timesi.ttf"),
        "bold_italic": Path(r"C:\Windows\Fonts\timesbi.ttf"),
    }
    if all(path.exists() for path in font_paths.values()):
        pdfmetrics.registerFont(TTFont("TNR", str(font_paths["regular"])))
        pdfmetrics.registerFont(TTFont("TNR-Bold", str(font_paths["bold"])))
        pdfmetrics.registerFont(TTFont("TNR-Italic", str(font_paths["italic"])))
        pdfmetrics.registerFont(TTFont("TNR-BoldItalic", str(font_paths["bold_italic"])))
        return {
            "regular": "TNR",
            "bold": "TNR-Bold",
            "italic": "TNR-Italic",
            "bold_italic": "TNR-BoldItalic",
        }
    return {
        "regular": "Times-Roman",
        "bold": "Times-Bold",
        "italic": "Times-Italic",
        "bold_italic": "Times-BoldItalic",
    }


def styles(fonts: dict[str, str]) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title_tr": ParagraphStyle(
            "TitleTR",
            parent=base["Title"],
            fontName=fonts["regular"],
            fontSize=20,
            leading=23,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "title_en": ParagraphStyle(
            "TitleEN",
            parent=base["Title"],
            fontName=fonts["regular"],
            fontSize=18,
            leading=21,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "author": ParagraphStyle(
            "Author",
            parent=base["Normal"],
            fontName=fonts["regular"],
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=0,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName=fonts["regular"],
            fontSize=9.0,
            leading=10.4,
            alignment=TA_JUSTIFY,
            firstLineIndent=10,
            spaceAfter=4,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["Normal"],
            fontName=fonts["bold"],
            fontSize=8.5,
            leading=9.7,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "keywords": ParagraphStyle(
            "Keywords",
            parent=base["Normal"],
            fontName=fonts["bold_italic"],
            fontSize=8.3,
            leading=9.5,
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading2"],
            fontName=fonts["regular"],
            fontSize=9.5,
            leading=11,
            alignment=TA_CENTER,
            spaceBefore=7,
            spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName=fonts["regular"],
            fontSize=7.2,
            leading=8.2,
            alignment=TA_CENTER,
            spaceAfter=5,
        ),
        "ref": ParagraphStyle(
            "Reference",
            parent=base["Normal"],
            fontName=fonts["regular"],
            fontSize=7.2,
            leading=8.2,
            alignment=TA_LEFT,
            firstLineIndent=-10,
            leftIndent=10,
            spaceAfter=2.5,
        ),
        "table": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName=fonts["regular"],
            fontSize=6.5,
            leading=7.2,
            alignment=TA_CENTER,
        ),
        "table_bold": ParagraphStyle(
            "TableCellBold",
            parent=base["Normal"],
            fontName=fonts["bold"],
            fontSize=6.5,
            leading=7.2,
            alignment=TA_CENTER,
        ),
    }


def load_status(results_dir: Path) -> dict[str, object]:
    status_path = results_dir / "run_status.json"
    if not status_path.exists():
        return {}
    return json.loads(status_path.read_text(encoding="utf-8"))


def load_aggregate(results_dir: Path) -> pd.DataFrame:
    aggregate_path = results_dir / "aggregate_by_engine.csv"
    if aggregate_path.exists():
        return pd.read_csv(aggregate_path)
    summary_path = results_dir / "experiment_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(summary_path)
    frame = frame[(frame["status"] == "ok") & frame["cer"].notna()]
    cols = ["cer", "wer", "diacritic_accuracy", "diacritic_error_count", "base_loss_count"]
    return frame.groupby(["engine", "stage"], as_index=False)[cols].mean()


def load_confusions(results_dir: Path, limit: int = 6) -> pd.DataFrame:
    path = results_dir / "diacritic_confusions.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    return (
        frame[frame["stage"] == "raw"]
        .groupby(["engine", "pair", "operation"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(limit)
    )


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("&", "&amp;"), style)


def section(title: str, roman: str, st: dict[str, ParagraphStyle]) -> list[Paragraph]:
    label = f"{roman}.  {title}" if roman else title
    return [paragraph(label, st["section"])]


def metric_table(aggregate: pd.DataFrame, st: dict[str, ParagraphStyle], col_width: float) -> Table:
    headers = ["Motor", "Aşama", "CER", "WER", "D-Acc", "D-Err"]
    rows = [[paragraph(h, st["table_bold"]) for h in headers]]
    if not aggregate.empty:
        ordered = aggregate.sort_values(["engine", "stage"])
        for _, row in ordered.iterrows():
            rows.append(
                [
                    paragraph(str(row["engine"]), st["table"]),
                    paragraph(str(row["stage"]), st["table"]),
                    paragraph(f"{row['cer']:.3f}", st["table"]),
                    paragraph(f"{row['wer']:.3f}", st["table"]),
                    paragraph(f"{row['diacritic_accuracy']:.3f}", st["table"]),
                    paragraph(f"{row['diacritic_error_count']:.1f}", st["table"]),
                ]
            )
    table = Table(rows, colWidths=[col_width * part for part in (0.18, 0.20, 0.14, 0.14, 0.16, 0.18)])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def correction_table(aggregate: pd.DataFrame, st: dict[str, ParagraphStyle], col_width: float) -> Table:
    headers = ["Motor", "ΔCER", "ΔWER", "ΔD-Err"]
    rows = [[paragraph(h, st["table_bold"]) for h in headers]]
    if not aggregate.empty:
        raw = aggregate[aggregate["stage"] == "raw"].set_index("engine")
        corrected = aggregate[aggregate["stage"] == "corrected"].set_index("engine")
        for engine in sorted(set(raw.index) & set(corrected.index)):
            rows.append(
                [
                    paragraph(engine, st["table"]),
                    paragraph(f"{raw.loc[engine, 'cer'] - corrected.loc[engine, 'cer']:.4f}", st["table"]),
                    paragraph(f"{raw.loc[engine, 'wer'] - corrected.loc[engine, 'wer']:.4f}", st["table"]),
                    paragraph(
                        f"{raw.loc[engine, 'diacritic_error_count'] - corrected.loc[engine, 'diacritic_error_count']:.2f}",
                        st["table"],
                    ),
                ]
            )
    table = Table(rows, colWidths=[col_width * part for part in (0.28, 0.24, 0.24, 0.24)])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def confusion_table(confusions: pd.DataFrame, st: dict[str, ParagraphStyle], col_width: float) -> Table:
    headers = ["Motor", "Karışım", "İşlem", "Sayı"]
    rows = [[paragraph(h, st["table_bold"]) for h in headers]]
    for _, row in confusions.iterrows():
        rows.append(
            [
                paragraph(str(row["engine"]), st["table"]),
                paragraph(str(row["pair"]), st["table"]),
                paragraph(str(row["operation"]), st["table"]),
                paragraph(str(int(row["count"])), st["table"]),
            ]
        )
    table = Table(rows, colWidths=[col_width * part for part in (0.22, 0.28, 0.28, 0.22)])
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def image_flowable(path: Path, col_width: float) -> Image | None:
    if not path.exists():
        return None
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        width, height = image.size
    render_width = col_width
    render_height = min(render_width * height / width, 5.2 * cm)
    return Image(str(path), width=render_width, height=render_height)


def draw_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("TNR" if "TNR" in pdfmetrics.getRegisteredFontNames() else "Times-Roman", 7)
    canvas.drawString(1.45 * cm, 0.55 * cm, "OCRTurk Türkçe OCR ve Diakritik Düzeltme")
    canvas.drawCentredString(A4[0] / 2, 0.55 * cm, str(doc.page))
    canvas.restoreState()


def build_story(
    st: dict[str, ParagraphStyle],
    results_dir: Path,
    author: str,
    affiliation: str,
    col_width: float,
) -> list[object]:
    aggregate = load_aggregate(results_dir)
    confusions = load_confusions(results_dir)
    status = load_status(results_dir)
    item_count = status.get("item_count", 180)
    ocrturk_repo_display = OCRTURK_REPO_URL.removesuffix(".git")

    story: list[object] = [
        paragraph("OCRTurk Üzerinde Türkçe OCR ve Diakritik Post-Correction", st["title_tr"]),
        paragraph("Turkish OCR and Diacritic Post-Correction on OCRTurk", st["title_en"]),
        paragraph(author, st["author"]),
        paragraph(affiliation, st["author"]),
        paragraph("Ankara, Türkiye", st["author"]),
        FrameBreak(),
        NextPageTemplate("TwoCol"),
        paragraph(
            "<i>Özetçe</i> - Bu çalışmada OCRTurk veri kümesi üzerinde Türkçe ham metin OCR performansı ve diakritik karakter hataları incelenmiştir. Tesseract ve EasyOCR, erişilebilir açık kaynak baseline olarak çalıştırılmış; çıktılar CER, WER, NED ve Türkçe karakter confusion ölçütleriyle değerlendirilmiştir. Ayrıca satır kırığı, karakter normalizasyonu ve sözlük destekli basit bir post-correction katmanı denenmiştir. Deneyler, Türkçe OCR değerlendirmesinde genel hata oranlarının yanında ç, ğ, ı, ö, ş ve ü gibi karakterlerin ayrı analiz edilmesi gerektiğini göstermektedir.",
            st["abstract"],
        ),
        paragraph(
            "<i>Anahtar Kelimeler</i> - Türkçe OCR, OCRTurk, diakritik düzeltme, post-correction, belge OCR",
            st["keywords"],
        ),
        paragraph(
            "<i>Abstract</i> - This paper reviews Turkish OCR challenges and presents a reproducible prototype experiment on OCRTurk. Tesseract and EasyOCR are evaluated as accessible baselines for raw text extraction, character and word error rates, and Turkish diacritic confusions. A lightweight rule and lexicon based post-correction layer is then applied. The results indicate that Turkish OCR should be analyzed not only through generic CER/WER scores, but also through language-specific diacritic errors.",
            st["abstract"],
        ),
        paragraph(
            "<i>Keywords</i> - Turkish OCR, OCRTurk, diacritic restoration, post-correction, document OCR",
            st["keywords"],
        ),
    ]

    sections: list[object] = []
    sections += section("GİRİŞ", "I", st)
    sections += [
        paragraph(
            "Optical Character Recognition (OCR), doküman görüntülerinden makine tarafından işlenebilir metin çıkarma problemidir. Türkçe için bu problem yalnızca Latin harflerini tanımakla sınırlı değildir; ç, ğ, ı, İ, ö, ş ve ü karakterleri, eklemeli dil yapısı ve PDF tabanlı doküman düzenleri hata davranışını doğrudan etkiler.",
            st["body"],
        ),
        paragraph(
            "Bu çalışmanın amacı yeni bir büyük OCR modeli eğitmek değil, OCRTurk üzerinde erişilebilir OCR motorlarının Türkçe karakter davranışını inceleyen kısa bir derleme ve deneysel ön çalışma sunmaktır. Böylece literatürdeki boşluk, tekrar üretilebilir bir deney hattı ve Türkçe karakter odaklı hata analizi aynı çalışma içinde raporlanmıştır.",
            st["body"],
        ),
        paragraph(
            "Türkçe metinlerde diakritik karakterler yalnızca yazım estetiği değildir; kelimenin anlamını, ek yapısını ve arama/eşleme işlemlerini etkiler. Örneğin 'şeker' ve 'seker', 'oldu' ve 'öldü' gibi biçimler OCR sonrası bilgi çıkarımı, sınıflandırma veya arşiv araması için farklı sonuçlar doğurabilir. Bu nedenle diakritik hataların ayrı bir hata ailesi olarak raporlanması gerekir.",
            st["body"],
        ),
        paragraph(
            "Bu raporda temel araştırma sorusu şudur: Tesseract ve EasyOCR gibi erişilebilir motorlar OCRTurk üzerinde Türkçe ham metin ve diakritik karakterlerde nasıl hata yapmaktadır, basit bir post-correction katmanı bu hataları hangi ölçüde azaltmaktadır? Bu soru, tam belge ayrıştırma yerine ham metin OCR ve dil-özel hata analiziyle sınırlandırılmıştır.",
            st["body"],
        ),
    ]
    sections += section("İLGİLİ ÇALIŞMALAR", "II", st)
    sections += [
        paragraph(
            "Türkçe OCR literatürü karakter tanıma, fiş ve belge görüntüleri, sahne metni ve el yazısı gibi parçalara ayrılmış durumdadır. Son sistematik değerlendirmeler, veri seti kıtlığı ve açık benchmark eksikliğinin Türkçe OCR için temel sınırlardan biri olduğunu vurgulamaktadır [3]. OCRTurk bu boşluğu azaltarak akademik doküman, tez, sunum ve akademik olmayan belge türlerinden oluşan 180 sayfalık bir benchmark sağlar [1], [2].",
            st["body"],
        ),
        paragraph(
            "Tesseract ve EasyOCR yaygın, erişilebilir ve tekrar üretilebilir OCR araçlarıdır [5], [6]. PaddleOCR ve modern vision-language yaklaşımları daha güçlü referanslar sunsa da bu çalışmada ana odak, herkesin çalıştırabileceği baseline motorlarda Türkçe diakritik hatalarının nasıl göründüğüdür [4], [7].",
            st["body"],
        ),
        paragraph(
            "OCRTurk'ün güçlü yönü, tek satır veya kelime görüntüleri yerine gerçek PDF sayfalarını kullanmasıdır. Bu sayfalarda font farklılıkları, tablo ve şekil bölgeleri, matematiksel ifade parçaları, başlık düzeni ve çok sütunlu yapı aynı anda bulunabilir. Bu durum, klasik OCR motorlarının yalnızca karakter tanıma değil, görüntü ön işleme ve okuma sırası açısından da zorlanmasına neden olur.",
            st["body"],
        ),
        paragraph(
            "Post-correction literatüründe iki ana yaklaşım görülür. İlk yaklaşım kural, sözlük ve n-gram tabanlıdır; hesaplama maliyeti düşük ve uygulanması kolaydır. İkinci yaklaşım BERTurk veya mT5 gibi bağlamsal dil modelleriyle aday düzeltme ya da seq2seq yeniden yazmadır [8], [9]. Bu çalışma, kapsamı sınırlı bir prototip için ilk yaklaşımı tercih eder ve dil modeli tabanlı düzeltmeyi gelecek çalışma olarak bırakır.",
            st["body"],
        ),
    ]
    sections += section("YÖNTEM", "III", st)
    sections += [
        paragraph(
            "OCRTurk klasörleri otomatik keşfedilmiş, PDF dosyaları PyMuPDF ile 300 DPI çözünürlükte PNG görüntülere çevrilmiş ve Markdown ground truth dosyaları düz metne indirgenmiştir. Her motorun ham OCR çıktısı normalize edilmiş, ardından aynı metrik hattından geçirilmiştir.",
            st["body"],
        ),
        paragraph(
            "Post-correction aşamasında önce satır kırığı ve sık karakter normalizasyonları uygulanmış, sonra leave-one-document-out yaklaşımla oluşturulan Türkçe diakritik sözlük kullanılmıştır. Bu tasarım, küçük benchmark üzerinde doğrudan ground truth sızıntısını azaltmak için her sayfanın kendi referans metnini aday sözlükten dışarıda bırakır.",
            st["body"],
        ),
        paragraph(
            "Değerlendirme hattı üç farklı düzeyde ölçüm üretir. CER karakter seviyesindeki edit uzaklığını, WER kelime seviyesindeki bozulmayı, NED ise normalleştirilmiş edit mesafesini verir. Bunlara ek olarak diakritik doğruluğu, yalnızca Türkçe karakter kümesine ait eşleşmeleri ölçerek genel karakter doğruluğunun sakladığı dil-özel hataları ayrıştırır.",
            st["body"],
        ),
        paragraph(
            "Diakritik confusion tablosu referans karakter ile OCR çıktısındaki karşılığını eşler. Bu tabloda ç/c, ğ/g, ı/i, İ/I/i, ö/o, ş/s ve ü/u gibi dönüşümler; silme, ekleme ve değiştirme işlemleriyle birlikte sayılır. Böylece bir motorun Türkçe karakterleri tamamen kaybedip kaybetmediği ya da yalnızca noktalama benzeri küçük hatalar üretip üretmediği görülebilir.",
            st["body"],
        ),
    ]
    sections += section("DENEY DÜZENİ", "IV", st)
    sections += [
        paragraph(
            f"Deneyler OCRTurk üzerinde {item_count} öğe için Tesseract 'tur' ve EasyOCR 'tr' motorlarıyla yürütülmüştür. Çalışma ayarı 300 DPI olup EasyOCR GPU desteğiyle çalıştırılmıştır. Ana metrikler CER, WER, NED, karakter doğruluğu, kelime doğruluğu, diakritik doğruluğu ve diakritik confusion sayılarıdır.",
            st["body"],
        ),
        paragraph(
            "Değerlendirme iki aşamalıdır: raw aşaması motorun doğrudan OCR çıktısını; corrected aşaması ise basit post-correction sonrası metni temsil eder. Bu ayrım, düşük maliyetli düzeltme katmanının hangi motorlarda yararlı olduğunu açıkça gösterir.",
            st["body"],
        ),
        paragraph(
            "Deney hattı veri indirme, PDF render, OCR, metrik hesaplama, grafik üretme ve rapor oluşturma adımlarını ayrık bileşenler halinde yürütür. Bu ayrım, aynı veri üzerinde yalnızca OCR motorunu, yalnızca post-correction katmanını veya yalnızca rapor üretimini yeniden çalıştırmayı mümkün kılar. OCRTurk verisi yeniden dağıtılmadığı için çalışmada yalnızca indirme yöntemi ve türetilen metrikler paylaşılır.",
            st["body"],
        ),
        paragraph(
            "Tesseract için Türkçe dil paketi kullanılmıştır. EasyOCR 'tr' okuyucusu GPU desteğiyle çalıştırılmıştır. PaddleOCR kurulumu opsiyonel bırakılmıştır; çünkü bu çalışmanın temel deney koşulu Tesseract ve EasyOCR baseline'larının tam OCRTurk üzerinde tamamlanmasıdır.",
            st["body"],
        ),
    ]
    sections += section("BULGULAR", "V", st)
    sections += [
        KeepTogether(
            [
                paragraph("Tablo I. Motor ve aşama bazlı ortalama metrikler", st["caption"]),
                metric_table(aggregate, st, col_width),
            ]
        ),
        paragraph(
            "Tablo I, Tesseract'ın bu prototip koşuda EasyOCR'dan daha düşük CER ve WER ürettiğini göstermektedir. EasyOCR için post-correction küçük ama pozitif bir CER/WER iyileşmesi sağlamış; Tesseract için ise aynı basit düzeltme katmanı genel metriklerde hafif gerilemeye yol açmıştır. Bu sonuç, correction katmanının motora duyarlı tasarlanması gerektiğini gösterir.",
            st["body"],
        ),
        paragraph(
            "Tesseract'ın daha düşük CER değeri, klasik OCR motorunun bu veri kümesindeki metin bölgelerini daha kararlı okuduğunu düşündürmektedir. Buna karşılık Tesseract çıktısında post-correction sonrası küçük bir kötüleşme oluşması, basit sözlük düzeltmesinin zaten doğru okunmuş karakterleri gereksiz yere değiştirebildiğini göstermektedir.",
            st["body"],
        ),
        paragraph(
            "EasyOCR tarafında raw ve corrected satırları arasındaki fark daha olumludur. Genel CER düşüşü küçük olsa da diakritik hata sayısındaki azalma daha belirgindir. Bu durum, post-correction katmanının özellikle diakritik karakter restorasyonu için yararlı olabileceğini, fakat genel kelime hatalarını tek başına çözmeye yetmediğini gösterir.",
            st["body"],
        ),
        KeepTogether(
            [
                paragraph("Tablo II. Post-correction etkisi: raw - corrected", st["caption"]),
                correction_table(aggregate, st, col_width),
            ]
        ),
    ]
    chart = image_flowable(results_dir / "plots" / "cer_by_engine_stage.png", col_width)
    if chart is not None:
        sections += [Spacer(1, 4), chart, paragraph("Şekil 1. CER değerlerinin motor ve aşamaya göre karşılaştırılması", st["caption"])]
    wer_chart = image_flowable(results_dir / "plots" / "wer_by_engine_stage.png", col_width)
    if wer_chart is not None:
        sections += [Spacer(1, 4), wer_chart, paragraph("Şekil 2. WER değerlerinin motor ve aşamaya göre karşılaştırılması", st["caption"])]
    sections += [
        paragraph(
            "Diakritik hata analizi, genel CER/WER skorlarının sakladığı Türkçe karakter davranışını görünür kılar. Özellikle noktalı/noktasız i ve ASCII tabanlı karakter düşmeleri, okunabilir görünen OCR çıktılarında bile Türkçe metin kalitesini düşürmektedir.",
            st["body"],
        )
    ]
    if not confusions.empty:
        sections += [
            KeepTogether(
                [
                    paragraph("Tablo III. En sık ham diakritik karışımları", st["caption"]),
                    confusion_table(confusions, st, col_width),
                ]
            )
        ]
    delta_chart = image_flowable(results_dir / "plots" / "postcorrection_cer_delta.png", col_width)
    if delta_chart is not None:
        sections += [Spacer(1, 4), delta_chart, paragraph("Şekil 3. Post-correction sonrası CER farkı", st["caption"])]
    dia_chart = image_flowable(results_dir / "plots" / "diacritic_error_count_by_engine_stage.png", col_width)
    if dia_chart is not None:
        sections += [
            Spacer(1, 4),
            dia_chart,
            paragraph("Şekil 4. Ortalama diakritik hata sayısı", st["caption"]),
            paragraph(
                "Şekil 4, EasyOCR çıktısında diakritik hata sayısının Tesseract'a göre daha yüksek olduğunu ve correction sonrasında azaldığını göstermektedir. Tesseract için ise correction katmanının diakritik hata sayısını artırması, sözlük tabanlı kararların motor güveni veya bağlam puanı ile desteklenmesi gerektiğini ortaya koyar.",
                st["body"],
            ),
        ]
    confusion_chart = image_flowable(results_dir / "plots" / "top_diacritic_confusions.png", col_width)
    if confusion_chart is not None:
        sections += [
            Spacer(1, 4),
            confusion_chart,
            paragraph("Şekil 5. En sık diakritik confusion örnekleri", st["caption"]),
        ]
    sections += section("TARTIŞMA", "VI", st)
    sections += [
        paragraph(
            "Sonuçlar, Türkçe OCR için tek bir genel skorun yeterli olmadığını göstermektedir. Tesseract genel metin çıkarımında daha güçlü görünürken, EasyOCR basit correction katmanından daha fazla yarar sağlamıştır. Buna karşılık Tesseract üzerinde aynı correction kuralları bazı doğru karakterleri bozabildiği için düzeltme kararları motor çıktısının hata profiline göre ayarlanmalıdır.",
            st["body"],
        ),
        paragraph(
            "Çalışmanın kapsamı ham metin OCR ve diakritik post-correction ile sınırlıdır. OCRTurk'ün tablo, denklem, figür ve Markdown yapısını kapsayan tam belge ayrıştırma görevi bu prototipin dışında bırakılmıştır. Gelecek çalışmada PaddleOCR ve BERTurk/mT5 tabanlı bağlamsal düzeltme eklenebilir [8], [9].",
            st["body"],
        ),
        paragraph(
            "Bu sınırlılık bilinçli bir kapsam daraltmasıdır. OCRTurk tam belge ayrıştırma için tasarlanmış olsa da Tesseract ve EasyOCR gibi araçlar doğrudan Markdown, tablo veya denklem yapısı üretmez. Bu nedenle bu çalışmada karşılaştırmanın adil kalması için ham metin çıkarımı ve diakritik doğruluk yüzeyi seçilmiştir.",
            st["body"],
        ),
        paragraph(
            "Sonuçların yorumlanmasında bir diğer nokta, post-correction katmanının modelden bağımsız olmamasıdır. Aynı kural seti bir motorda hataları azaltırken başka bir motorda hata ekleyebilir. Daha güçlü bir düzeltici için motor güven skorları, kelime sıklığı, bağlam modeli ve Türkçe morfolojik analiz birlikte kullanılmalıdır.",
            st["body"],
        ),
        paragraph(
            "Buna rağmen deneysel tasarımın pratik değeri vardır. Veri indirme, OCR koşusu ve metrik üretimi aynı hat üzerinden tekrar üretilebilir; sonuç tabloları hem genel metrikleri hem de diakritik hata ayrıntılarını içerir. Bu yapı, daha sonra PaddleOCR veya dil modeli tabanlı correction eklemek için doğrudan genişletilebilir.",
            st["body"],
        ),
    ]
    sections += section("SONUÇ", "VII", st)
    sections += [
        paragraph(
            "Bu derleme ve deneysel ön çalışma, OCRTurk üzerinde Türkçe OCR performansını erişilebilir motorlar, tekrar üretilebilir yöntem ve diakritik odaklı hata analiziyle raporlamıştır. Üretilen sonuçlar, genel OCR skorlarının tek başına Türkçe karakter davranışını açıklamak için yeterli olmadığını göstermektedir.",
            st["body"],
        ),
        paragraph(
            "Deneyler, Tesseract'ın genel CER/WER bakımından daha güçlü baseline olduğunu; EasyOCR'ın ise basit post-correction katmanından diakritik hata sayısı açısından daha çok yarar sağladığını göstermiştir. Bu bulgu, Türkçe OCR uygulamalarında düzeltme katmanının tek tip değil, motor ve hata profiline göre uyarlanmış olması gerektiğini destekler.",
            st["body"],
        ),
    ]
    sections += section("TEKRAR ÜRETİLEBİLİRLİK", "VIII", st)
    sections += [
        paragraph(
            "Deney hattı Python betikleriyle tekrar çalıştırılabilir durumdadır. OCRTurk verisi indirildikten sonra Tesseract ve EasyOCR motorları 300 DPI render edilmiş sayfalar üzerinde koşturulmuştur. Metrik tabloları ve grafikler aynı deney hattı kullanılarak yeniden üretilebilir.",
            st["body"],
        ),
        paragraph(
            "Temel tekrar üretim akışı; bağımlılıkların kurulması, run_experiments.py ile OCRTurk üzerinde deneyin çalıştırılması ve build_conference_report_pdf.py ile bildiri PDF'inin üretilmesi adımlarından oluşur. Aynı veri ve motor kurulumuyla rapordaki tablo ve şekiller yeniden elde edilebilir.",
            st["body"],
        ),
    ]
    sections += section("GEÇERLİLİK TEHDİTLERİ", "IX", st)
    sections += [
        paragraph(
            "İç geçerlilik açısından en önemli risk, post-correction sözlüğünün ground truth metinlerden türetilmiş olmasıdır. Bu risk, her sayfa için kendi referans metnini aday sözlükten çıkaran leave-one-document-out tasarımla azaltılmıştır. Yine de sözlük tabanlı düzeltmenin gerçek dünyadaki OCR çıktılarında aynı etkiyi üretmesi garanti değildir.",
            st["body"],
        ),
        paragraph(
            "Dış geçerlilik açısından sonuçlar yalnızca OCRTurk'ün 180 sayfalık veri dağılımı için doğrudan yorumlanmalıdır. OCRTurk farklı belge türleri içerse de taranmış düşük kaliteli arşiv belgeleri, el yazısı dokümanlar veya kamera görüntüleri bu deneyin kapsamına girmez. Bu nedenle sonuçlar Türkçe OCR geneli için ilk gösterge, fakat nihai hüküm olarak değerlendirilmemelidir.",
            st["body"],
        ),
        paragraph(
            "Yapısal geçerlilik açısından bu çalışma OCRTurk'ün tam Markdown üretim görevini ölçmez. Tablolar, denklemler, şekiller ve okuma sırası hataları metin skoruna dolaylı olarak yansır, fakat ayrı yapısal skorlanmaz. Bu tercih, örnek baseline motorların yetenekleriyle uyumlu ve yönetilebilir bir kapsam oluşturmak için yapılmıştır.",
            st["body"],
        ),
    ]
    sections += section("MATERYALLER VE VERİ ERİŞİMİ", "X", st)
    sections += [
        paragraph(
            f"Deney betikleri, metrik hesaplama modülleri ve rapor üretim kodu açık kaynak çalışma deposunda yer alır: {PROJECT_REPO_URL}. OCR koşuları run_experiments.py ile yürütülür; diakritik analiz ve düzeltme bileşenleri ayrı modüller halinde düzenlenmiştir. Üretilen CSV tabloları ve grafikler aynı deney hattından elde edilir.",
            st["body"],
        ),
        paragraph(
            f"OCRTurk verisi bu çalışma içinde yeniden dağıtılmamaktadır; veri, metunlp/ocrturk deposundan indirilmektedir ({ocrturk_repo_display}). Bu tercih, veri kaynağının izlenebilir kalmasını ve deneylerin aynı açık kaynak veri deposu üzerinden tekrar üretilebilmesini sağlar.",
            st["body"],
        ),
    ]
    sections += section("GELECEK ÇALIŞMA", "XI", st)
    sections += [
        paragraph(
            "Bir sonraki teknik genişletme, PaddleOCR'ın aynı sonuç tablosuna üçüncü motor olarak eklenmesidir. PaddleOCR daha modern bir OCR hattı sunduğu için Tesseract ve EasyOCR baseline'larıyla karşılaştırıldığında özellikle belge düzeni ve karmaşık sayfa bölgelerinde daha güçlü bir referans sağlayabilir.",
            st["body"],
        ),
        paragraph(
            "İkinci genişletme, basit sözlük düzeltmesi yerine bağlam duyarlı bir reranking katmanı kullanmaktır. BERTurk tabanlı aday seçimi, yalnızca kelimenin sözlükte bulunmasını değil, cümle içindeki olasılığını da dikkate alabilir. Böyle bir yaklaşım, Tesseract'ta görülen gereksiz düzeltme sorununu azaltabilir.",
            st["body"],
        ),
        paragraph(
            "Üçüncü genişletme, OCRTurk'ün tam belge ayrıştırma hedefini yeniden kapsama almaktır. Bu durumda yalnızca ham metin değil, Markdown başlıkları, tablolar, denklemler, figür açıklamaları ve okuma sırası da değerlendirilecektir. Böyle bir çalışma daha uzun süre ve daha güçlü belge parsing araçları gerektirir.",
            st["body"],
        ),
        paragraph(
            "Son olarak, insan değerlendirmesiyle desteklenen küçük bir hata örnekleri bölümü eklenebilir. CER/WER metrikleri sayısal karşılaştırma sağlar; ancak Türkçe diakritik hatalarının anlam üzerindeki etkisi, seçilmiş örnek cümleler üzerinden nitel olarak tartışıldığında daha anlaşılır hale gelir.",
            st["body"],
        ),
    ]
    sections += section("KAYNAKLAR", "", st)
    for index, (title, url) in enumerate(REFERENCES, start=1):
        sections.append(KeepTogether([paragraph(f"[{index}] {title} {url}", st["ref"])]))
    story.extend(sections)
    return story


def build_pdf(results_dir: Path, output_path: Path, author: str, affiliation: str) -> None:
    fonts = register_fonts()
    st = styles(fonts)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_width, page_height = A4
    margin_x = 1.35 * cm
    margin_bottom = 1.25 * cm
    margin_top = 1.35 * cm
    gap = 0.45 * cm
    col_width = (page_width - 2 * margin_x - gap) / 2

    first_title = Frame(margin_x, page_height - 7.1 * cm, page_width - 2 * margin_x, 5.6 * cm, id="title", showBoundary=0)
    first_left = Frame(margin_x, margin_bottom, col_width, page_height - 8.1 * cm, id="first_left", showBoundary=0)
    first_right = Frame(margin_x + col_width + gap, margin_bottom, col_width, page_height - 8.1 * cm, id="first_right", showBoundary=0)
    later_left = Frame(
        margin_x,
        margin_bottom,
        col_width,
        page_height - margin_top - margin_bottom,
        id="later_left",
        showBoundary=0,
    )
    later_right = Frame(
        margin_x + col_width + gap,
        margin_bottom,
        col_width,
        page_height - margin_top - margin_bottom,
        id="later_right",
        showBoundary=0,
    )

    doc = BaseDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_top,
        bottomMargin=margin_bottom,
    )
    doc.addPageTemplates(
        [
            PageTemplate(id="First", frames=[first_title, first_left, first_right], onPage=draw_footer),
            PageTemplate(id="TwoCol", frames=[later_left, later_right], onPage=draw_footer),
        ]
    )
    doc.build(build_story(st, results_dir, author, affiliation, col_width))


def main() -> None:
    args = parse_args()
    build_pdf(args.results_dir, args.output, args.author, args.affiliation)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
