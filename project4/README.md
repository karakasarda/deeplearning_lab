# Project 4: Turkish Toxic Language Detection

Bu proje, `Overfit-GM/turkish-toxic-language` veri seti uzerinde Turkce toksik dil tespitini deneysel olarak inceler. Ana fikir yalnizca en yuksek skoru almak degil; dengesiz sinif dagilimi, birlestirilmis veri kaynaklari, kalibrasyon ve `toxic / other` karar esigi seciminin sonuca etkisini olcmektir.

## Research question

**Birlestirilmis ve dengesiz Turkce toksisite verisinde on isleme, sinif agirlikli lineer taban model ve post-hoc temperature scaling, macro-F1 ve moderasyon karar guvenilirligini nasil degistirir?**

## Dataset

- Kaynak: <https://huggingface.co/datasets/Overfit-GM/turkish-toxic-language>
- Kolonlar: `text`, `target`, `source`, `is_toxic`
- Etiketler: `OTHER`, `PROFANITY`, `INSULT`, `RACIST`, `SEXIST`
- Operasyonel ikili karar: `OTHER` disindaki tum etiketler `toxic`

Kod, veriyi Hugging Face uzerinden `data/raw/` altina indirir, bos/uyumsuz satirlari temizler, ayni metnin birden fazla etiket aldigi cakismali ornekleri cikarir ve exact duplicate temizligi yapar. Split islemi sabit `seed=42` ile stratified `70/15/15` olarak yapilir.

## Methods

### Linear baseline

`run_experiments.py` uc metin varyantini dener:

- `raw`: yalniz whitespace duzeltmesi
- `light`: URL, kullanici adi, hashtag ve tekrar eden karakter normalizasyonu
- `light_masked`: `light` uzerine sinirli kufur/profanity maskeleme

Her varyant icin ortak model:

- TF-IDF word n-gram: `1-2`
- TF-IDF char n-gram: `3-5`
- Sinif agirlikli lineer log-loss classifier
- Validation set uzerinde temperature scaling
- Validation set uzerinde `toxic / other` threshold sweep

### Transformer extension

`train_transformer.py`, ayni temizleme ve split mantigiyla BERTurk, Turkish ELECTRA, XLM-R, TurkishBERTweet ve ConvBERTurk modellerini fine-tune etmek icin hazirlandi. Bu kisim `transformers`, `datasets`, `accelerate`, `sentencepiece` ve `safetensors` gerektirir.

Transformer model registry:

| Key | Model |
| --- | --- |
| `berturk` | `dbmdz/bert-base-turkish-cased` |
| `electra` | `dbmdz/electra-base-turkish-cased-discriminator` |
| `xlm_r` | `FacebookAI/xlm-roberta-base` |
| `turkishbertweet` | `VRLLab/TurkishBERTweet` |
| `convberturk` | `dbmdz/convbert-base-turkish-cased` |

### Limited hyperparameter search

`hpo_transformer.py`, BERTurk uzerinde 6 sabit trial ile kisa HPO kosar. Arama `sample_size=12000`, `epochs=0.5` ile yapilir ve secilen ayar diger transformer modellerine tasinir. Secim kurali: once validation macro-F1, 0.003 icindeki kosullarda daha dusuk classwise-ECE, sonra daha kisa runtime.

### Full transformer suite

`run_full_suite.py`, HPO'yu, 5 transformer `light` kosusunu, ek `berturk_raw` ablation kosusunu, final raporu ve sunum asset uretimini sirayla calistirir. OOM durumunda batch size `4`, gradient accumulation `4` ile tekrar dener.

## Current results

Tam veri kosusu `77,800` ornekte, stratified `70/15/15` split ile calistirildi. Exact duplicate veya cakismali metin etiketi bulunmadi. En iyi lineer sonuc `light` on isleme varyantindan geldi.

| Variant | Model | Temp. | Test Acc | Test Macro-F1 | Test Macro PR-AUC | Top-label ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `light` | SGD log-loss | 1.05 | 0.8820 | 0.7969 | 0.8592 | 0.0083 |
| `raw` | SGD log-loss | 1.05 | 0.8766 | 0.7866 | 0.8452 | 0.0082 |
| `light_masked` | SGD log-loss | 1.15 | 0.8608 | 0.7642 | 0.8308 | 0.0087 |

En iyi modelin sinif bazli F1 sonuclari:

| Label | F1 | Support |
| --- | ---: | ---: |
| `OTHER` | 0.9352 | 5649 |
| `PROFANITY` | 0.8971 | 2738 |
| `INSULT` | 0.7482 | 1617 |
| `RACIST` | 0.8228 | 1525 |
| `SEXIST` | 0.5811 | 141 |

`toxic / other` karar katmani icin validation set uzerinde secilen iki politika test setinde su sonucu verdi:

| Policy | Threshold | Precision | Recall | F1 | Flagged rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `best_f1` | 0.44 | 0.9394 | 0.9395 | 0.9395 | 0.5160 |
| `recall_floor_90` | 0.64 | 0.9654 | 0.8980 | 0.9305 | 0.4799 |

Bu sonuc raporun ana iddiasini destekliyor: genel accuracy yuksek gorunse de `SEXIST` sinifi dusuk destek nedeniyle belirgin sekilde daha zor kaliyor; bu yuzden macro-F1 ve sinif bazli hata analizi accuracy'den daha bilgilendirici.

## Run

Mevcut GPU/Python ortami ile:

```powershell
cd "D:\Ankara Üniversitesi\Ankara Üni 2025-2026\Bahar\Derin Öğrenme\project4"
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" run_experiments.py
```

Hizli smoke test:

```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" run_experiments.py --sample-size 8000 --skip-plots
```

Transformer bagimliliklarini kurduktan sonra:

```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" -m pip install -r requirements.txt
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" train_transformer.py --model-name dbmdz/bert-base-turkish-cased --epochs 2
```

Smoke test:

```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" train_transformer.py --model-key berturk --run-name smoke_berturk --preprocess light --sample-size 2000 --epochs 0.1 --output-dir .\results\smoke_transformer\smoke_berturk
```

Tam kosu:

```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" run_full_suite.py --skip-existing
```

Final rapor ve sunum assetleri tek basina yenilemek icin:

```powershell
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" build_final_report.py
& "$env:USERPROFILE\dl-gpu-py313\Scripts\python.exe" build_presentation.py
$env:NODE_PATH='C:\Users\Arda Karakaş\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
& 'C:\Users\Arda Karakaş\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' .\presentation\build_deck.mjs
```

## Outputs

Ana calisma `results/` altina sunlari uretir:

- `audit_summary.json`
- `experiment_summary.csv`
- `report.md`
- `best_per_class_report.csv`
- `best_confusion_matrix.csv`
- `threshold_sweep_val.csv`
- `threshold_policies_test.csv`
- `variant_*/high_confidence_errors.csv`
- `plots/class_distribution.png`
- `variant_*/confusion_matrix.png`
- `variant_*/per_class_f1.png`
- `variant_*/threshold_sweep_val.png`
- `variant_*/reliability_test.png`
- `final_model_comparison.csv`
- `detailed_report.md`
- `hpo/hpo_summary.csv`
- `hpo/hpo_best.json`
- `transformers/*/metrics.json`
- `output/Project4_Turkish_Toxic_Language_Detection.pptx`

## Report source

Proje fikrinin kapsam analizi icin [`deep-research-report (1).md`](deep-research-report%20(1).md) dosyasi kullanildi.
