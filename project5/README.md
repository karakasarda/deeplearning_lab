# Project 5: OCRTurk Turkish OCR Review and Prototype

Bu proje, YZM304 5. proje tanımındaki **derleme (review) + prototip deney** beklentisine göre hazırlanır. Ana çıktı sunum değil, `ornek_bilidiriler/` klasöründeki örnekler gibi **iki sütunlu IEEE bildiri formatında PDF rapor** ve bu raporu destekleyen tekrar üretilebilir OCRTurk prototipidir.

Araştırma sorusu:

> Tesseract, EasyOCR ve opsiyonel PaddleOCR, OCRTurk üzerinde Türkçe ham metin ve diakritik karakterlerde nasıl hata yapıyor; basit post-correction bu hataları ne kadar azaltıyor?

## Scope

- Ana teslim: örnek bildirilerle uyumlu IEEE-style PDF report + prototype code/results
- Zorunlu baseline: `tesseract`, `easyocr`
- Opsiyonel baseline: `paddleocr`
- Test/smoke motoru: `mock`
- Kapsam dışı: tablo, denklem, figür, HTML/LaTeX/Markdown yapısal parsing skoru
- Kapsam dışı: sunum/PPTX üretimi
- Veri kaynağı: <https://github.com/metunlp/ocrturk>
- Makale: <https://aclanthology.org/2026.sigturk-1.16/>

OCRTurk verisi repo içinde yeniden dağıtılmaz. `data/raw/` altına indirilir ve `.gitignore` kapsamındadır.

## Deliverables

- `output/pdf/Project5_OCRTurk_IEEE_Format_Report.pdf`: örnek bildirilerle uyumlu 4 sayfalık iki sütunlu PDF bildiri
- `output/Project5_OCRTurk_IEEE_Review_Report.md`: PDF raporu destekleyen Markdown rapor taslağı
- `results/final/experiment_summary.csv`: final deney metrikleri
- `results/final/diacritic_confusions.csv`: Türkçe diakritik hata tablosu
- `results/final/plots/*.png`: raporda kullanılacak grafikler
- `src/` ve `run_experiments.py`: prototip deney kodu

Sunum dosyası üretilmez ve teslim paketine dahil edilmez.

## Setup

Python bağımlılıkları:

```powershell
cd "D:\Ankara Üniversitesi\Ankara Üni 2025-2026\Bahar\Derin Öğrenme\deeplearning_lab\project5"
python -m pip install -r requirements.txt
```

Tesseract için Windows binary ve Türkçe language pack ayrıca kurulmalıdır. Kod, `tesseract` PATH'te değilse standart `C:\Program Files\Tesseract-OCR\tesseract.exe` kurulumunu da dener. Kurulumdan sonra şu komutlar çalışmalıdır:

```powershell
tesseract --version
tesseract --list-langs
```

`--list-langs` çıktısında `tur` görünmelidir.

Sistem `tessdata` klasörüne yazma izni yoksa `tur.traineddata` dosyası `tools/tessdata/` altına konabilir. Proje yolu Türkçe karakter içerdiğinde Tesseract sorun çıkardığı için kod bu dosyayı çalışma sırasında ASCII yollu `C:\ocrturk_tessdata` cache klasöründen kullanır.

## Smoke Test

Gerçek OCR motoru kurmadan metrik, diakritik analiz, correction ve rapor hattını test etmek için:

```powershell
python run_experiments.py --dataset-root data/demo/ocrturk --make-demo --engines mock --skip-render --force-ocr --results-dir results/smoke
python build_ieee_review_report.py --results-dir results/smoke
```

Unit test:

```powershell
python -m unittest discover -s tests
```

## Full Run

OCRTurk verisini indirip Tesseract ve EasyOCR ile koşmak için:

```powershell
python run_experiments.py --download-data --engines tesseract easyocr --dpi 300 --results-dir results/final
python build_ieee_review_report.py --results-dir results/final
python build_conference_report_pdf.py --results-dir results/final
```

PaddleOCR eklemek için:

```powershell
python run_experiments.py --download-data --engines tesseract easyocr paddleocr --dpi 300 --gpu --results-dir results/final
```

Küçük gerçek veri smoke testi:

```powershell
python run_experiments.py --download-data --engines tesseract easyocr --limit 5 --dpi 300 --results-dir results/smoke_real
python build_ieee_review_report.py --results-dir results/smoke_real
```

## Outputs

Ana çıktılar:

- `results/final/experiment_summary.csv`: sayfa/motor/aşama bazlı CER, WER, NED ve diakritik skorlar
- `results/final/diacritic_confusions.csv`: Türkçe karakter confusion satırları
- `results/final/aggregate_by_engine.csv`: motor ve aşama bazlı ortalamalar
- `results/final/run_status.json`: motor uygunluğu, hata ve runtime özeti
- `results/final/plots/*.png`: metrik ve confusion grafikleri
- `results/final/ieee_review_report.md`: final deney klasöründeki rapor kopyası
- `output/Project5_OCRTurk_IEEE_Review_Report.md`: teslim raporu
- `output/pdf/Project5_OCRTurk_IEEE_Format_Report.pdf`: örnek bildiri formatındaki nihai PDF teslim dosyası

OCR metin çıktıları `results/outputs/` altına yazılır ve `.gitignore` kapsamındadır.

## Method Notes

Ground truth Markdown dosyaları plain text'e indirgenir. Görseller, tablolar, HTML ve LaTeX işaretleri ham metin skorundan çıkarılır. OCR çıktısı önce normalize edilir, sonra iki aşamada değerlendirilir:

- `raw`: OCR motorundan gelen normalize çıktı
- `corrected`: satır kırığı düzeltmesi, sınırlı OCR kuralları ve leave-one-document-out Türkçe diakritik sözlük düzeltmesi

Post-correction sözlüğü her sayfa için kendi ground truth metnini dışarıda bırakır. Bu, küçük benchmark üzerinde doğrudan leakage riskini azaltır.

## Current Environment Notes

Bu makinede Python ve CUDA erişimi var. Render için varsayılan araç PyMuPDF olduğu için Poppler zorunlu değildir. Tesseract binary standart Windows kurulum yolundan bulunur; Türkçe dil dosyası yerel `tools/tessdata` / `C:\ocrturk_tessdata` akışıyla kullanılabilir.
