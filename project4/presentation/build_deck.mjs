import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const require = createRequire(import.meta.url);
const tool = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  text,
  image,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  fr,
} = tool;

const data = JSON.parse(fs.readFileSync(path.join('presentation', 'deck_data.json'), 'utf8'));
const OUT = 'output';
fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(path.join('presentation', 'previews'), { recursive: true });

const W = 1920;
const H = 1080;
const ink = '#17212B';
const muted = '#5B6673';
const accent = '#0E7C7B';
const gold = '#C8892B';
const rose = '#B23A48';
const ruleColor = '#D5DDD8';

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

function addSlide(root) {
  const slide = presentation.slides.add();
  slide.compose(root, { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 });
  return slide;
}

function titleStack(title, subtitle = '') {
  return column({ name: 'title-stack', width: fill, height: hug, gap: 10 }, [
    text(title, { name: 'slide-title', width: fill, height: hug, style: { fontSize: 42, bold: true, color: ink } }),
    subtitle ? text(subtitle, { name: 'slide-subtitle', width: wrap(1500), height: fixed(56), style: { fontSize: 21, color: muted } }) : null,
    rule({ name: 'title-rule', width: fixed(220), stroke: accent, weight: 4 }),
  ].filter(Boolean));
}

function bodyText(value, width = fill, fontSize = 21, color = ink) {
  return text(value, { width, height: hug, style: { fontSize, color } });
}

function smallLabel(value, color = muted) {
  return text(value, { width: fill, height: hug, style: { fontSize: 15, color, bold: true } });
}

function metric(label, value, color = ink, note = '') {
  return column({ name: `metric-${label}`, width: fill, height: hug, gap: 4 }, [
    text(String(value), { width: fill, height: hug, style: { fontSize: 48, bold: true, color } }),
    text(label, { width: fill, height: hug, style: { fontSize: 17, color: muted, bold: true } }),
    note ? text(note, { width: fill, height: hug, style: { fontSize: 14, color: muted } }) : null,
  ].filter(Boolean));
}

function bulletLines(items, fontSize = 21) {
  return column({ name: 'bullets', width: fill, height: hug, gap: 11 },
    items.map((item, idx) => text(`• ${item}`, { name: `bullet-${idx}`, width: fill, height: hug, style: { fontSize, color: ink } })));
}

function rowsTable(rows, columns, options = {}) {
  const fontSize = options.fontSize ?? 17;
  const headerSize = options.headerSize ?? 15;
  const rowGap = options.rowGap ?? 8;
  const header = row({ name: 'table-header', width: fill, height: hug, gap: 14 },
    columns.map(c => text(c.label, { width: fixed(c.w), height: hug, style: { fontSize: headerSize, bold: true, color: muted } })));
  const body = rows.map((r, idx) => row({ name: `table-row-${idx}`, width: fill, height: hug, gap: 14 },
    columns.map(c => text(String(r[c.key] ?? ''), {
      width: fixed(c.w),
      height: hug,
      style: { fontSize, color: idx === 0 && options.emphasizeFirst !== false ? accent : ink, bold: idx === 0 && options.emphasizeFirst !== false },
    }))));
  return column({ name: 'table', width: fill, height: hug, gap: rowGap }, [
    header,
    rule({ width: fill, stroke: ruleColor, weight: 2 }),
    ...body,
  ]);
}

function imageSource(imagePath) {
  if (!imagePath || !fs.existsSync(imagePath)) {
    return null;
  }
  const ext = path.extname(imagePath).toLowerCase();
  const contentType = ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : 'image/png';
  const dataUrl = `data:${contentType};base64,${fs.readFileSync(imagePath).toString('base64')}`;
  return { dataUrl, contentType };
}

function imageOrText(name, imagePath, fallback) {
  const src = imageSource(imagePath);
  if (src) {
    return image({ name, ...src, width: fill, height: fill, fit: 'contain', alt: name });
  }
  return text(fallback, { name, width: fill, height: hug, style: { fontSize: 24, color: muted } });
}

function takeaway(title, lines) {
  return column({ name: 'takeaway', width: fill, height: hug, gap: 10 }, [
    smallLabel(title, accent),
    bulletLines(lines, 19),
  ]);
}

addSlide(column({ name: 'cover', width: fill, height: fill, padding: { x: 96, y: 74 }, gap: 30 }, [
  text('PROJECT 4 / YZM304', { name: 'kicker', width: fill, height: hug, style: { fontSize: 22, color: accent, bold: true } }),
  text(data.title, { name: 'cover-title', width: wrap(1360), height: hug, style: { fontSize: 62, bold: true, color: ink } }),
  text(data.subtitle, { name: 'cover-subtitle', width: wrap(1320), height: hug, style: { fontSize: 27, color: muted } }),
  bodyText(`Ana iddia: ${data.claim}`, wrap(1420), 22, ink),
  row({ name: 'cover-metrics', width: fill, height: hug, gap: 74 }, [
    metric('best macro-F1', data.bestMacroF1, accent, data.bestRun),
    metric('best accuracy', data.bestAccuracy, ink, 'destekleyici metrik'),
    metric('top-label ECE', data.bestEce, gold, 'kalibrasyon göstergesi'),
    metric('baseline lift', `+${data.macroLift}`, rose, 'macro-F1 farkı'),
  ]),
  row({ name: 'cover-context', width: fill, height: hug, gap: 56 }, [
    bodyText(`${data.datasetRows} satır`, fixed(220), 20, muted),
    bodyText('5 sınıf', fixed(160), 20, muted),
    bodyText('3 baseline + 6 transformer koşusu', fixed(430), 20, muted),
    bodyText('temperature scaling + threshold sweep', fixed(500), 20, muted),
  ]),
]));

addSlide(row({ name: 'problem', width: fill, height: fill, padding: { x: 78, y: 58 }, gap: 60 }, [
  column({ name: 'problem-left', width: fixed(760), height: fill, gap: 26 }, [
    titleStack('Accuracy tek başına moderasyon kalitesini anlatmıyor', 'Sınıf dengesizliği, aynı accuracy değerinin farklı hata profilleri saklamasına neden olur.'),
    text('Bu veri setinde başarıyı tek bir skorla değil, karar davranışıyla okuyoruz.', { width: fill, height: hug, style: { fontSize: 43, bold: true, color: ink } }),
    bulletLines([
      'Macro-F1 küçük sınıfları accuracy kadar kolay ezmez.',
      'ECE model güveninin ne kadar kalibre olduğunu gösterir.',
      'Threshold seçimi moderation tarafında precision/recall dengesini değiştirir.',
    ]),
  ]),
  column({ name: 'problem-right', width: fill, height: fill, gap: 22 }, [
    smallLabel('Kritik metriklerin anlamı'),
    rowsTable([
      { metric: 'Accuracy', meaning: 'Tüm test örneklerinde doğru oranı', risk: 'Büyük sınıflar baskın gelir' },
      { metric: 'Macro-F1', meaning: 'Sınıf F1 ortalaması', risk: 'Ana seçim metriği' },
      { metric: 'PR-AUC', meaning: 'Sınıf olasılık ayrımı', risk: 'Dengesiz veri için güçlü okuma' },
      { metric: 'ECE', meaning: 'Güven skoru kalitesi', risk: 'Düşük daha iyi' },
      { metric: 'Threshold', meaning: 'Toxic/other karar eşiği', risk: 'Operasyonel politika' },
    ], [
      { key: 'metric', label: 'Metrik', w: 150 },
      { key: 'meaning', label: 'Ne ölçer?', w: 380 },
      { key: 'risk', label: 'Bu projedeki rolü', w: 330 },
    ], { fontSize: 18 }),
    takeaway('Bu slayttaki okuma', [
      `En iyi model accuracy ${data.bestAccuracy} üretse de ana iddia macro-F1 ${data.bestMacroF1} ve sınıf bazlı hata analizine dayanıyor.`,
      '`SEXIST` test desteği 141; bu yüzden sınıf bazlı tablo zorunlu.',
    ]),
  ]),
]));

addSlide(column({ name: 'data-audit', width: fill, height: fill, padding: { x: 72, y: 52 }, gap: 20 }, [
  titleStack('Veri denetimi: büyük ama dengesiz ve heterojen', `${data.datasetRows} satır; medyan uzunluk ${data.textMedian} karakter, p95 uzunluk ${data.textP95} karakter.`),
  row({ name: 'data-body', width: fill, height: fill, gap: 28 }, [
    column({ name: 'data-images', width: fixed(780), height: fill, gap: 18 }, [
      imageOrText('class-distribution', data.classDistributionImage, 'Class distribution plot not available'),
      imageOrText('source-distribution', data.sourceDistributionImage, 'Source distribution plot not available'),
    ]),
    column({ name: 'data-tables', width: fill, height: fill, gap: 26 }, [
      rowsTable(data.classRows, [
        { key: 'label', label: 'Sınıf', w: 180 },
        { key: 'count', label: 'Satır', w: 130 },
        { key: 'share', label: 'Pay', w: 90 },
      ], { fontSize: 18 }),
      rowsTable(data.sourceRows, [
        { key: 'source', label: 'Kaynak', w: 180 },
        { key: 'count', label: 'Satır', w: 130 },
        { key: 'share', label: 'Pay', w: 90 },
      ], { fontSize: 18, emphasizeFirst: false }),
      takeaway('Neden önemli?', [
        'Kaynak heterojenliği domain shift riskini artırır.',
        '`SEXIST` tüm verinin yaklaşık %1.2’si; macro-F1 bu yüzden ana başarı metriği.',
      ]),
    ]),
  ]),
]));

addSlide(column({ name: 'pipeline', width: fill, height: fill, padding: { x: 78, y: 58 }, gap: 28 }, [
  titleStack('Deney hattı sabit split ve ortak metriklerle kuruldu', 'Aynı split, aynı label sırası ve aynı post-hoc değerlendirme bütün modellerde korunur.'),
  row({ name: 'pipeline-body', width: fill, height: fill, gap: 44 }, [
    rowsTable([
      { step: 'Split', detail: 'Stratified 70/15/15, seed=42', why: 'Test set karşılaştırması sabit' },
      { step: 'Label order', detail: 'OTHER, PROFANITY, INSULT, RACIST, SEXIST', why: 'Tüm CSV/plotlarda aynı sıra' },
      { step: 'Preprocess', detail: 'raw / light / light_masked', why: 'Baseline ablation + transformer light' },
      { step: 'Baseline', detail: 'TF-IDF + SGD log-loss', why: 'Güçlü, hızlı alt sınır' },
      { step: 'Transformers', detail: 'BERTurk, ELECTRA, XLM-R, BERTweet, ConvBERTurk', why: 'Türkçe ve multilingual karşılaştırma' },
      { step: 'Calibration', detail: 'Validation logits ile temperature scaling', why: 'Test metrikleri kalibre olasılıktan' },
    ], [
      { key: 'step', label: 'Aşama', w: 190 },
      { key: 'detail', label: 'Uygulama', w: 560 },
      { key: 'why', label: 'Gerekçe', w: 460 },
    ], { fontSize: 18 }),
    column({ name: 'pipeline-notes', width: fixed(420), height: hug, gap: 24 }, [
      smallLabel('Eğitim ayarı'),
      bulletLines([
        '1 epoch, max_length=128',
        'batch=8, grad_acc=2',
        'weight_decay=0.01',
        'warmup_ratio=0.06',
        'class_weight açık',
        'OOM fallback: batch=4, grad_acc=4',
      ], 20),
      smallLabel('Üretilen artefactler'),
      bulletLines([
        'metrics.json, per_class_report.csv',
        'confusion_matrix.csv/png',
        'reliability ve threshold plotları',
        'final_model_comparison.csv',
      ], 20),
    ]),
  ]),
]));

addSlide(column({ name: 'hpo', width: fill, height: fill, padding: { x: 78, y: 58 }, gap: 24 }, [
  titleStack('HPO sınırlı ama savunulabilir tutuldu', 'BERTurk üzerinde 6 kısa trial; seçilen ayar diğer transformer koşularına taşındı.'),
  row({ name: 'hpo-body', width: fill, height: fill, gap: 44 }, [
    rowsTable(data.hpoRows, [
      { key: 'trial', label: '#', w: 60 },
      { key: 'lr', label: 'LR', w: 100 },
      { key: 'cw', label: 'Class wt.', w: 110 },
      { key: 'valF1', label: 'Val macro-F1', w: 150 },
      { key: 'ece', label: 'Classwise ECE', w: 150 },
      { key: 'time', label: 'Runtime', w: 100 },
    ], { fontSize: 18 }),
    column({ name: 'hpo-notes', width: fixed(520), height: hug, gap: 20 }, [
      smallLabel('Seçilen ayar'),
      rowsTable([
        { key: 'trial', value: data.hpo.trial ?? 'n/a' },
        { key: 'learning_rate', value: data.hpo.learning_rate_display },
        { key: 'weight_decay', value: data.hpo.weight_decay ?? 'n/a' },
        { key: 'warmup_ratio', value: data.hpo.warmup_ratio ?? 'n/a' },
        { key: 'class_weight', value: String(data.hpo.class_weight ?? 'n/a') },
      ], [
        { key: 'key', label: 'Parametre', w: 210 },
        { key: 'value', label: 'Değer', w: 180 },
      ], { fontSize: 18 }),
      takeaway('Seçim kuralı', [
        'Önce validation macro-F1 maksimize edildi.',
        '0.003 içindeki eşitliklerde classwise ECE ve runtime’a bakıldı.',
        'Bu tam Optuna araması değil; raporda limited HPO olarak yazıldı.',
      ]),
    ]),
  ]),
]));

addSlide(column({ name: 'comparison', width: fill, height: fill, padding: { x: 72, y: 52 }, gap: 24 }, [
  titleStack('Model karşılaştırması macro-F1 üzerinden okunmalı', `En iyi koşu ${data.bestRun}; baseline’a göre macro-F1 artışı +${data.macroLift}.`),
  row({ name: 'comparison-body', width: fill, height: fill, gap: 36 }, [
    imageOrText('comparison-image', data.comparisonImage, 'Comparison plot not available'),
    column({ name: 'comparison-table', width: fixed(760), height: fill, gap: 22 }, [
      rowsTable(data.modelRows.slice(0, 7), [
        { key: 'model', label: 'Model', w: 300 },
        { key: 'macroF1', label: 'F1', w: 85 },
        { key: 'prAuc', label: 'PR-AUC', w: 105 },
        { key: 'ece', label: 'ECE', w: 85 },
      ], { fontSize: 17 }),
      takeaway('Okuma', [
        'BERTurk light en yüksek macro-F1 değerini verdi.',
        'BERTurk raw PR-AUC’ta çok güçlü ama macro-F1’de light koşusunun az gerisinde.',
        'XLM-R multilingual kontrol olarak geride kaldı; Türkçe odaklı modeller avantajlı.',
      ]),
    ]),
  ]),
]));

addSlide(column({ name: 'confusion', width: fill, height: fill, padding: { x: 66, y: 48 }, gap: 18 }, [
  titleStack('Confusion matrix hatanın nerede toplandığını gösteriyor', 'Raw sayımlar hata büyüklüğünü, normalized görünüm sınıf içi oranı gösterir.'),
  row({ name: 'confusion-body', width: fill, height: fill, gap: 28 }, [
    column({ name: 'confusion-images', width: fixed(1120), height: fill, gap: 14 }, [
      row({ name: 'matrix-row', width: fill, height: fill, gap: 20 }, [
        imageOrText('best-confusion', data.bestConfusionImage, 'Confusion matrix not available'),
        imageOrText('best-confusion-normalized', data.bestConfusionNormalizedImage, 'Normalized confusion matrix not available'),
      ]),
    ]),
    column({ name: 'confusion-notes', width: fill, height: hug, gap: 18 }, [
      smallLabel('En büyük karışmalar'),
      rowsTable(data.confusionPairs, [
        { key: 'pair', label: 'True -> Pred', w: 210 },
        { key: 'count', label: 'Adet', w: 80 },
        { key: 'rate', label: 'Oran', w: 80 },
      ], { fontSize: 17 }),
      takeaway('Yorum', [
        '`INSULT` örneklerinin bir bölümü `OTHER` veya `PROFANITY` ile karışıyor.',
        '`SEXIST` recall yüksek olsa da düşük destek yüzünden precision kırılgan.',
        '`RACIST` diagonal oranı güçlü; hata daha çok komşu toksik sınıflara dağılıyor.',
      ]),
    ]),
  ]),
]));

addSlide(column({ name: 'per-class', width: fill, height: fill, padding: { x: 72, y: 52 }, gap: 22 }, [
  titleStack('Sınıf bazlı sonuçlar accuracy’nin sakladığı riski açıyor', '`SEXIST`, `INSULT` ve `RACIST` sınıfları moderasyon açısından ayrı yorumlanmalı.'),
  row({ name: 'per-class-body', width: fill, height: fill, gap: 38 }, [
    imageOrText('per-class-image', data.bestPerClassImage, 'Per-class plot not available'),
    column({ name: 'per-class-table', width: fixed(760), height: hug, gap: 22 }, [
      rowsTable(data.perClassRows, [
        { key: 'label', label: 'Label', w: 160 },
        { key: 'precision', label: 'Prec.', w: 95 },
        { key: 'recall', label: 'Rec.', w: 95 },
        { key: 'f1', label: 'F1', w: 90 },
        { key: 'support', label: 'Support', w: 110 },
      ], { fontSize: 18 }),
      takeaway('Sınıf yorumu', [
        '`OTHER` ve `PROFANITY` çok güçlü; büyük destek etkisi var.',
        '`INSULT` daha semantik sınırda kaldığı için F1 düşüyor.',
        '`SEXIST` testte yalnızca 141 örnek; sonuç iyi ama güven aralığı geniş olabilir.',
      ]),
    ]),
  ]),
]));

addSlide(column({ name: 'calibration', width: fill, height: fill, padding: { x: 72, y: 52 }, gap: 22 }, [
  titleStack('Kalibrasyon: yüksek skorun güvenilir olasılık üretmesi gerekiyor', 'Temperature scaling validation logits üzerinde öğrenildi; test metrikleri kalibre olasılıklardan raporlandı.'),
  row({ name: 'calibration-body', width: fill, height: fill, gap: 42 }, [
    imageOrText('reliability-plot', data.bestReliabilityImage, 'Reliability plot not available'),
    column({ name: 'calibration-table', width: fixed(760), height: hug, gap: 22 }, [
      rowsTable(data.calibrationRows, [
        { key: 'model', label: 'Model', w: 300 },
        { key: 'topECE', label: 'Top ECE', w: 110 },
        { key: 'classECE', label: 'Class ECE', w: 120 },
        { key: 'logLoss', label: 'Log loss', w: 110 },
      ], { fontSize: 17 }),
      takeaway('Okuma', [
        'Top-label ECE tek başına yeterli değil; classwise ECE küçük sınıf güvenini ayrıca gösterir.',
        'BERTurk light hem macro-F1 hem classwise ECE açısından en dengeli koşu.',
      ]),
    ]),
  ]),
]));

addSlide(column({ name: 'thresholds', width: fill, height: fill, padding: { x: 72, y: 52 }, gap: 22 }, [
  titleStack('Threshold seçimi moderasyon davranışını değiştiriyor', '`best_f1` daha dengeli, `recall_floor_90` daha muhafazakar yüksek precision politikasıdır.'),
  row({ name: 'threshold-body', width: fill, height: fill, gap: 42 }, [
    imageOrText('threshold-curve', data.bestThresholdImage, 'Threshold plot not available'),
    column({ name: 'threshold-table', width: fixed(760), height: hug, gap: 24 }, [
      rowsTable(data.policyRows, [
        { key: 'policy', label: 'Policy', w: 210 },
        { key: 'threshold', label: 'Thr.', w: 80 },
        { key: 'precision', label: 'Prec.', w: 95 },
        { key: 'recall', label: 'Rec.', w: 95 },
        { key: 'f1', label: 'F1', w: 95 },
        { key: 'flagged', label: 'Flagged', w: 95 },
      ], { fontSize: 17 }),
      takeaway('Karar etkisi', [
        '`best_f1` yaklaşık yarı veri akışını toxic olarak işaretler.',
        '`recall_floor_90` precision’ı artırır ama recall ve F1’den ödün verir.',
        'Bu nedenle deployment için skor değil politika raporlanmalıdır.',
      ]),
    ]),
  ]),
]));

addSlide(column({ name: 'landing', width: fill, height: fill, padding: { x: 86, y: 62 }, gap: 26 }, [
  titleStack('Sonuç: raporlanması gereken şey skor değil, güvenilir karar davranışı', 'Final paket model karşılaştırmasını, sınıf bazlı hata analizini, kalibrasyonu ve threshold politikasını birlikte veriyor.'),
  row({ name: 'landing-body', width: fill, height: fill, gap: 60 }, [
    column({ name: 'landing-findings', width: fixed(780), height: hug, gap: 18 }, [
      smallLabel('Güçlü bulgular'),
      bulletLines([
        `BERTurk light test macro-F1 ${data.bestMacroF1} ile en iyi genel dengeyi verdi.`,
        `TF-IDF baseline güçlü kaldı ama transformer en iyi baseline’dan +${data.macroLift} macro-F1 ileride.`,
        '`SEXIST` sınıfı düşük destek yüzünden hala en kırılgan sınıf.',
        'Kalibrasyon ve threshold sweep sonuçları moderasyon kararının modele ek olarak politika gerektirdiğini gösterdi.',
      ], 22),
    ]),
    column({ name: 'landing-package', width: fill, height: hug, gap: 18 }, [
      smallLabel('Teslim paketi'),
      rowsTable([
        { item: 'Kod', file: 'train_transformer.py, run_full_suite.py' },
        { item: 'Final sonuç', file: 'results/final_model_comparison.csv' },
        { item: 'Detaylı rapor', file: 'results/detailed_report.md' },
        { item: 'Sunum', file: 'output/Project4_Turkish_Toxic_Language_Detection.pptx' },
        { item: 'QA render', file: 'presentation/previews/final_pptx/*.png' },
      ], [
        { key: 'item', label: 'Parça', w: 180 },
        { key: 'file', label: 'Dosya', w: 560 },
      ], { fontSize: 18, emphasizeFirst: false }),
      smallLabel('Sınırlılıklar'),
      bulletLines([
        'HPO yalnız BERTurk üzerinde sınırlı aramayla yapıldı.',
        'Kısa epoch tercihi laptop GPU runtime sınırından kaynaklandı.',
        'Veri pseudo-label ve kaynak birleşimi içerdiği için altın standart olarak yorumlanmamalı.',
      ], 20),
    ]),
  ]),
]));

const appendix = data.appendixMatrices.slice(0, 6);
addSlide(column({ name: 'appendix', width: fill, height: fill, padding: { x: 52, y: 44 }, gap: 14 }, [
  titleStack('Appendix: tüm ana transformer confusion matrix görünümleri', 'Küçük çoklu görünüm, ana anlatının ek kanıtıdır; baseline matrix’leri detaylı raporda yer alır.'),
  grid({ name: 'appendix-grid', width: fill, height: fill, columns: [fr(1), fr(1), fr(1)], rows: [fr(1), fr(1)], columnGap: 24, rowGap: 14 },
    appendix.map((item, idx) => column({ name: `appendix-cell-${idx}`, width: fill, height: fill, gap: 6 }, [
      text(item.run, { width: fill, height: hug, style: { fontSize: 17, bold: true, color: ink } }),
      imageOrText(`appendix-image-${idx}`, item.path, 'missing'),
    ]))),
]));

const pptxBlob = await PresentationFile.exportPptx(presentation);
await pptxBlob.save(path.join(OUT, 'Project4_Turkish_Toxic_Language_Detection.pptx'));
console.log(path.join(OUT, 'Project4_Turkish_Toxic_Language_Detection.pptx'));
