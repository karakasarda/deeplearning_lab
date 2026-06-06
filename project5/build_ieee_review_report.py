from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.config import OCRTURK_PAPER_URL, OCRTURK_REPO_URL, OUTPUT_DIR, PROJECT_REPO_URL, RESULTS_DIR
from src.reporting import aggregate_summary, markdown_table, plot_confusion_heatmap, plot_metric_bars


REFERENCES = [
    {
        "id": 1,
        "text": "D. Yılmaz, E. A. Munis, C. Toraman, S. K. Köse, B. Aktaş, M. C. Baytekin, and B. K. Görür, \"OCRTurk: A Comprehensive OCR Benchmark for Turkish,\" in Proceedings of SIGTURK 2026, pp. 197-208, 2026, doi: 10.18653/v1/2026.sigturk-1.16.",
        "url": OCRTURK_PAPER_URL,
    },
    {
        "id": 2,
        "text": "METU NLP, \"OCRTurk dataset repository,\" GitHub, 2026.",
        "url": "https://github.com/metunlp/ocrturk",
    },
    {
        "id": 3,
        "text": "M. G. Öztürk, D. Ö. Şahin, and E. Kılıç, \"Turkish Optical Character Recognition Under the Lens: A Systematic Review of Language-Specific Challenges, Dataset Scarcity, and Open-Source Limitations,\" IEEE Access, vol. 13, pp. 168977-168997, 2025, doi: 10.1109/ACCESS.2025.3614147.",
        "url": "https://doi.org/10.1109/ACCESS.2025.3614147",
    },
    {
        "id": 4,
        "text": "Y. Yılmaz, E. G. Hanoğlu, A. G. Özkan, and K. Öztoprak, \"Benchmarking OCR and Vision-Language Models for Turkish Text Recognition: A Comprehensive Evaluation Using Synthetic Data,\" Research Square preprint, 2025, doi: 10.21203/rs.3.rs-7797886/v1.",
        "url": "https://doi.org/10.21203/rs.3.rs-7797886/v1",
    },
    {
        "id": 5,
        "text": "Tesseract OCR contributors, \"Tesseract OCR documentation,\" 2026.",
        "url": "https://tesseract-ocr.github.io/",
    },
    {
        "id": 6,
        "text": "Jaided AI, \"EasyOCR: Ready-to-use OCR with 80+ supported languages,\" GitHub repository, 2024.",
        "url": "https://github.com/JaidedAI/EasyOCR",
    },
    {
        "id": 7,
        "text": "PaddlePaddle, \"PaddleOCR: multilingual OCR and document parsing toolkit,\" GitHub repository, 2026.",
        "url": "https://github.com/PaddlePaddle/PaddleOCR",
    },
    {
        "id": 8,
        "text": "MDZ Digital Library team, \"dbmdz/bert-base-turkish-cased: BERTurk model card,\" Hugging Face, 2026.",
        "url": "https://huggingface.co/dbmdz/bert-base-turkish-cased",
    },
    {
        "id": 9,
        "text": "L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel, \"mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer,\" in Proceedings of NAACL-HLT 2021, pp. 483-498, 2021, doi: 10.18653/v1/2021.naacl-main.41.",
        "url": "https://aclanthology.org/2021.naacl-main.41/",
    },
    {
        "id": 10,
        "text": "C. Sayallar, A. Sayar, and N. Babalık, \"An OCR Engine for Printed Receipt Images using Deep Learning Techniques,\" International Journal of Advanced Computer Science and Applications, vol. 14, no. 2, pp. 833-840, 2023, doi: 10.14569/IJACSA.2023.0140295.",
        "url": "https://doi.org/10.14569/IJACSA.2023.0140295",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Project 5 IEEE-style review/prototype report")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--title", default="OCRTurk Üzerinde Türkçe OCR ve Diakritik Post-Correction: Derleme ve Prototip Çalışma")
    return parser.parse_args()


def read_status(results_dir: Path) -> dict[str, object]:
    path = results_dir / "run_status.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def top_confusions(results_dir: Path, limit: int = 8) -> pd.DataFrame:
    path = results_dir / "diacritic_confusions.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    return (
        frame.groupby(["engine", "stage", "pair", "operation"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(limit)
    )


def correction_delta(aggregate: pd.DataFrame) -> pd.DataFrame:
    if aggregate.empty:
        return pd.DataFrame()
    raw = aggregate[aggregate["stage"] == "raw"].set_index("engine")
    corrected = aggregate[aggregate["stage"] == "corrected"].set_index("engine")
    rows = []
    for engine in sorted(set(raw.index) & set(corrected.index)):
        rows.append(
            {
                "engine": engine,
                "cer_delta": raw.loc[engine, "cer"] - corrected.loc[engine, "cer"],
                "wer_delta": raw.loc[engine, "wer"] - corrected.loc[engine, "wer"],
                "diacritic_error_delta": raw.loc[engine, "diacritic_error_count"]
                - corrected.loc[engine, "diacritic_error_count"],
            }
        )
    return pd.DataFrame(rows)


def run_context(results_dir: Path) -> str:
    status = read_status(results_dir)
    if not status:
        return "Henüz deney koşusu bulunmamaktadır; rapordaki deney kısmı prototip yöntemi ve çalıştırma planı olarak sunulmuştur."
    engines = ", ".join(status.get("engines", []))
    item_count = status.get("item_count", "unknown")
    run_config = status.get("run_config", {})
    config_bits = []
    if run_config:
        config_bits.append(f"DPI={run_config.get('dpi', 'unknown')}")
        config_bits.append(f"GPU={'yes' if run_config.get('gpu') else 'no'}")
    unavailable = []
    for name, info in status.get("availability", {}).items():
        if not info.get("available"):
            unavailable.append(f"{name}: {info.get('reason')}")
    suffix = f" Çalışmayan motorlar: {'; '.join(unavailable)}." if unavailable else ""
    config_text = f" Çalışma ayarı: {', '.join(config_bits)}." if config_bits else ""
    return f"Bu rapor oluşturulurken `{results_dir}` altında {item_count} sayfa/öğe için {engines} koşusu kullanılmıştır.{config_text}{suffix}"


def build_results_section(results_dir: Path, figure_prefix: str | None = None) -> list[str]:
    summary_path = results_dir / "experiment_summary.csv"
    aggregate = aggregate_summary(summary_path)
    plots = plot_metric_bars(summary_path, results_dir / "plots")
    confusion_plot = plot_confusion_heatmap(results_dir / "diacritic_confusions.csv", results_dir / "plots")
    if confusion_plot:
        plots.append(confusion_plot)

    if aggregate.empty:
        return [
            "## VI. Prototype Results and Error Analysis",
            "",
            "Bu aşamada nihai OCRTurk koşusu bulunmadığı için sonuç bölümü deney protokolünü ve beklenen analiz biçimini raporlar. Final koşuda `experiment_summary.csv`, `diacritic_confusions.csv` ve grafikler aynı komut zinciriyle üretilecektir.",
            "",
            "```powershell",
            "python run_experiments.py --download-data --engines tesseract easyocr --dpi 300 --results-dir results/final",
            "python build_ieee_review_report.py --results-dir results/final",
            "```",
        ]

    delta = correction_delta(aggregate)
    confusions = top_confusions(results_dir)
    lines = [
        "## VI. Prototype Results and Error Analysis",
        "",
        run_context(results_dir),
        "",
        "### A. Aggregate OCR Metrics",
        markdown_table(aggregate.round(4), max_rows=12),
        "",
        "CER ve WER genel metin çıkarımını, diakritik doğruluğu ise Türkçe karakter duyarlılığını ölçer. Bu iki okuma birlikte kullanıldığında bir motorun yalnızca okunabilir metin üretip üretmediği değil, Türkçe'ye özgü karakterleri ne kadar koruduğu da görünür hale gelir.",
        "",
        "### B. Post-Correction Effect",
        markdown_table(delta.round(4), max_rows=8) if not delta.empty else "_Correction delta üretilemedi._",
        "",
        "Pozitif delta, correction sonrasında hata oranının düştüğünü gösterir. Bu çalışmada düzeltme katmanı bilinçli olarak basit tutulmuştur; amaç büyük dil modeliyle maksimum skor almak değil, düşük maliyetli Türkçe diakritik düzeltmenin prototip değerini ölçmektir.",
        "",
        "### C. Diacritic Confusions",
        markdown_table(confusions, max_rows=8) if not confusions.empty else "_Diakritik confusion satırı bulunamadı._",
    ]
    if plots:
        lines.extend(["", "### D. Generated Figures"])
        for path in plots[:5]:
            if figure_prefix is None:
                rel = path.relative_to(results_dir).as_posix() if results_dir in path.parents else path.as_posix()
            else:
                rel = f"{figure_prefix.rstrip('/')}/{path.name}"
            lines.append(f"![{path.stem}]({rel})")
    return lines


def references_section() -> list[str]:
    lines = ["## References", ""]
    for ref in REFERENCES:
        lines.append(f"[{ref['id']}] {ref['text']} {ref['url']}")
    return lines


def build_report(results_dir: Path, title: str, figure_prefix: str | None = None) -> str:
    results_lines = build_results_section(results_dir, figure_prefix=figure_prefix)
    lines = [
        f"# {title}",
        "",
        "**Çıktı türü:** IEEE/IMRAD tarzı 4-5 sayfalık derleme + deneysel ön çalışma  \n**Çalışma odağı:** Literatür taraması, araştırma boşluğu ve orta ölçekli deney hattı",
        "",
        "## Abstract",
        "Türkçe OCR çalışmaları uzun süre karakter, fiş, sahne metni veya el yazısı gibi parçalı alt problemlere odaklanmıştır. OCRTurk, gerçek Türkçe doküman sayfalarını belge ayrıştırma bağlamında sunarak bu boşluğu azaltan yeni bir benchmark'tır. Bu rapor, OCRTurk'ü tam yapısal parsing yerine ham metin OCR ve Türkçe diakritik hata analizi açısından ele alan bir derleme ve prototip çalışma sunar. Literatür taraması, Türkçe OCR'de veri kıtlığı, açık kaynak benchmark eksikliği ve diakritik karakterlerin hata analizinde ayrı ele alınması gerektiğini göstermektedir. Prototip bölümünde Tesseract ve EasyOCR erişilebilir baseline olarak seçilmiş, PaddleOCR opsiyonel güçlü referans olarak konumlandırılmıştır. Çıktılar CER, WER, NED ve diakritik confusion tablolarıyla değerlendirilir; ardından kural tabanlı ve sözlük destekli düşük maliyetli post-correction uygulanır.",
        "",
        "**Keywords:** Turkish OCR, OCRTurk, diacritic restoration, post-correction, document OCR, deep learning review",
        "",
        "## I. Introduction",
        "Optical Character Recognition (OCR), doküman görüntülerinden metin çıkarma problemidir; ancak Türkçe için problem yalnızca Latin harflerini okumaktan ibaret değildir. `ç`, `ğ`, `ı`, `İ`, `ö`, `ş`, `ü` gibi karakterler, zengin eklemeli yapı, satır kırılmaları ve karmaşık akademik doküman düzenleri OCR hatalarını doğrudan etkiler. İngilizce ağırlıklı eğitim ve benchmark kültürü, Türkçe karakter hatalarını genellikle genel CER/WER içinde eritmektedir.",
        "",
        "Bu çalışmanın amacı yeni bir büyük OCR modeli eğitmek değildir. Amaç, OCRTurk etrafında literatürdeki boşluğu görünür kılmak, erişilebilir OCR motorları için tekrar üretilebilir bir deney hattı kurmak ve Türkçe diakritik hatalarını ayrı bir analiz nesnesi haline getirmektir. Bu yaklaşım; kapsamlı ama kısa bir derleme, araştırma boşluğu ve orta ölçekli deneysel ön çalışmayı aynı raporda birleştirir.",
        "",
        "## II. Literature Review",
        "Türkçe OCR literatürü son yıllarda büyüse de sistematik derlemeler alanın hâlâ parçalı olduğunu göstermektedir [3]. Çalışmaların bir bölümü font veya karakter tanımaya, bir bölümü fiş ve bankacılık dokümanlarına, bir bölümü sahne metni ve el yazısına odaklanır. Bu çalışmalar yararlı olmakla birlikte gerçek PDF sayfalarında tablo, denklem, şekil ve çok sütunlu düzen içeren akademik dokümanları tek bir benchmark altında toplamaz.",
        "",
        "OCRTurk bu noktada önemli bir dönüm noktasıdır. Benchmark; akademik dokümanlar, akademik olmayan dokümanlar, tezler ve sunumlar gibi farklı türlerden 180 sayfa içerir ve ground truth'u Markdown yapısı üzerinden temsil eder [1], [2]. Buna karşılık Tesseract ve EasyOCR gibi yaygın araçlar doğal olarak tam belge parsing çıktısı üretmez; daha çok metin tanıma ve satır/paragraf düzeyinde OCR için uygundur [5], [6]. PaddleOCR ise daha güçlü belge OCR bileşenleriyle OCRTurk makalesindeki modern sistem ailesine daha yakındır [7].",
        "",
        "2025 tarihli Türkçe OCR ve vision-language model benchmark çalışmaları, modern VLM'lerin bazı Türkçe metin tanıma görevlerinde klasik OCR sistemlerini geçebildiğini göstermiştir [4]. Ancak bu tür çalışmalar çoğu zaman sentetik veya kelime/satır düzeyi görsellere dayanır. OCRTurk'ün değeri, gerçek doküman sayfalarında düzen, kaynak türü ve zorluk seviyesini birlikte taşımasıdır. Bu nedenle OCRTurk üzerinde erişilebilir baseline'ların diakritik karakter davranışını ölçmek, küçük ama özgün bir araştırma boşluğunu hedefler.",
        "",
        "OCR post-correction tarafında iki ana çizgi vardır: düşük maliyetli kural/sözlük yaklaşımları ve bağlam duyarlı dil modeli yaklaşımları. BERTurk gibi Türkçe encoder modelleri aday seçme ve yeniden sıralama için, mT5 gibi text-to-text modeller ise daha kapsamlı seq2seq düzeltme için kullanılabilir [8], [9]. Bu rapordaki prototip, veri ve süre sınırları nedeniyle önce düşük maliyetli düzeltmeye odaklanır.",
        "",
        "## III. Research Gap",
        "OCRTurk makalesi güçlü modern OCR sistemlerini karşılaştırır; ancak Tesseract ve EasyOCR gibi kolay erişilebilir baseline'lar için Türkçe diakritik odaklı ayrıntılı bir analiz sunmaz. Diğer Türkçe OCR çalışmaları ise çoğunlukla OCRTurk kapsamındaki gerçek doküman parsing bağlamına oturmaz. Bu çalışma şu boşluğu hedefler: OCRTurk üzerinde ham metin OCR performansını, Türkçe karakter confusion analizi ve basit post-correction etkisiyle birlikte raporlamak.",
        "",
        "## IV. Prototype Method",
        "Prototip dört aşamadan oluşur. İlk aşamada OCRTurk veri klasörleri keşfedilir; PDF, Markdown ground truth ve varsa `source.json` metadata eşleştirilir. İkinci aşamada PDF sayfaları PyMuPDF ile sabit DPI değerinde PNG'ye render edilir. Üçüncü aşamada OCR motorları aynı görüntüler üzerinde çalıştırılır. Dördüncü aşamada çıktı metinleri normalize edilir, ground truth Markdown plain text'e indirgenir ve metrikler hesaplanır.",
        "",
        "Post-correction katmanı iki seviyede tasarlanmıştır. Önce satır kırığı, ligature ve sık OCR karakter karışıklıkları normalize edilir. Daha sonra leave-one-document-out mantığıyla ground truth metinlerden diakritik aday sözlüğü çıkarılır. Her sayfa için kendi ground truth metni sözlük adaylarından dışarıda bırakılır; bu tercih küçük benchmark üzerinde doğrudan veri sızıntısını azaltır.",
        "",
        "## V. Experimental Setup",
        "Zorunlu baseline'lar Tesseract `tur` ve EasyOCR `tr` olarak belirlenmiştir. PaddleOCR, kurulum ve çalışma zamanı izin verdiğinde opsiyonel güçlü referans olarak eklenir. Ana metrikler CER, WER, NED, character accuracy, word accuracy, diacritic accuracy, base-loss count ve diacritic confusion count'tur. `ç->c`, `ğ->g`, `ı->i`, `ö->o`, `ş->s`, `ü->u` gibi dönüşümler ayrı satırlarda raporlanır.",
        "",
    ]
    lines.extend(results_lines)
    lines.extend(
        [
            "",
            "## VII. Discussion",
            "Bu derleme ve prototipin temel katkısı, Türkçe OCR performansını yalnızca genel hata oranı olarak değil, dilin karakter sistemine özgü hata davranışı olarak okumaktır. Genel CER düşük olsa bile diakritik karakterlerin sistematik olarak ASCII tabanlarına düşmesi Türkçe metin kalitesini ciddi biçimde bozar. Bu yüzden post-correction başarısı yalnızca WER düşüşüyle değil, diakritik confusion tablosundaki azalmayla da değerlendirilmelidir.",
            "",
            "Çalışmanın sınırlılığı, yapısal OCRTurk görevini tam olarak çözmemesidir. Tablo, denklem ve figür parsing bu raporun kapsamı dışındadır. Ayrıca dil modeli tabanlı correction yalnızca gelecek çalışma olarak konumlandırılmıştır; mevcut prototip düşük maliyetli ve tekrar üretilebilir bir ilk deney düzeyi hedefler.",
            "",
            "## VIII. Conclusion",
            "OCRTurk, Türkçe belge OCR çalışmaları için güçlü bir araştırma zemini sunar. Bu rapor, alan literatürünü kısa bir derleme olarak özetleyip erişilebilir OCR motorları üzerinde diakritik odaklı bir prototip deney hattı kurmuştur. Sonraki teknik genişletme, PaddleOCR ve dil modeli tabanlı correction katmanlarını eklemek ve bulguları konferans bildirisi formatında daha ayrıntılı tartışmaktır.",
            "",
            "## Reproducibility",
            f"Çalışmanın açık kaynak deposu: {PROJECT_REPO_URL}",
            "",
            "```powershell",
            "python -m pip install -r requirements.txt",
            "python run_experiments.py --download-data --engines tesseract easyocr --dpi 300 --results-dir results/final",
            "python build_ieee_review_report.py --results-dir results/final",
            "```",
            "",
        ]
    )
    lines.extend(references_section())
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / "ieee_review_report.md"
    output_path = args.output_dir / "Project5_OCRTurk_IEEE_Review_Report.md"
    results_report = build_report(args.results_dir, args.title)
    output_figure_prefix = Path(
        os.path.relpath((args.results_dir / "plots").resolve(), args.output_dir.resolve())
    ).as_posix()
    output_report = build_report(args.results_dir, args.title, figure_prefix=output_figure_prefix)
    results_path.write_text(results_report, encoding="utf-8")
    output_path.write_text(output_report, encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
