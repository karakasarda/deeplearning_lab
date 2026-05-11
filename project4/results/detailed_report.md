# Project 4 Detailed Report

## 1. Dataset audit and protocol
- Raw rows: `77800`
- Clean rows: `77800`
- Class distribution: `{'OTHER': 37663, 'PROFANITY': 18252, 'INSULT': 10777, 'RACIST': 10163, 'SEXIST': 945}`
- Split: stratified `70/15/15`, seed `42`.
- Primary task: five-class single-label classification.
- Operational task: `OTHER` vs toxic threshold selection.

## 2. Hyperparameter optimization
| trial | run_name | status | learning_rate | weight_decay | warmup_ratio | class_weight | val_macro_f1 | val_classwise_ece | val_top_label_ece | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hpo_berturk_trial_01 | completed | 2e-05 | 0.0100 | 0.0600 | True | 0.6587 | 0.0263 | 0.0268 | 80.3566 |
| 2 | hpo_berturk_trial_02 | completed | 3e-05 | 0.0100 | 0.0600 | True | 0.7267 | 0.0279 | 0.0185 | 79.6696 |
| 3 | hpo_berturk_trial_03 | completed | 1e-05 | 0.0100 | 0.0600 | True | 0.5607 | 0.0387 | 0.0718 | 79.4765 |
| 4 | hpo_berturk_trial_04 | completed | 2e-05 | 0.0000 | 0.0000 | True | 0.6545 | 0.0275 | 0.0211 | 78.8768 |
| 5 | hpo_berturk_trial_05 | completed | 2e-05 | 0.0100 | 0.1000 | True | 0.6670 | 0.0244 | 0.0219 | 79.7838 |
| 6 | hpo_berturk_trial_06 | completed | 2e-05 | 0.0100 | 0.0600 | False | 0.6670 | 0.0152 | 0.0192 | 77.4717 |

Selected config: `{'trial': 2, 'run_name': 'hpo_berturk_trial_02', 'learning_rate': 3e-05, 'weight_decay': 0.01, 'warmup_ratio': 0.06, 'class_weight': True, 'selection_rule': 'max val_macro_f1; within 0.003 prefer lower classwise_ece, then runtime', 'val_macro_f1': 0.7267337522343423, 'val_classwise_ece': 0.02791064968984574}`

## 3. Model comparison
| run_name | family | model | preprocess | test_accuracy | test_balanced_accuracy | test_macro_f1 | test_weighted_f1 | test_macro_pr_auc | test_log_loss | test_top_label_ece | test_classwise_ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| berturk_light | transformer | BERTurk | light | 0.9423 | 0.9131 | 0.8997 | 0.9425 | 0.9456 | 0.1708 | 0.0089 | 0.0078 |
| berturk_raw | transformer | BERTurk | raw | 0.9434 | 0.9129 | 0.8981 | 0.9438 | 0.9510 | 0.1659 | 0.0111 | 0.0094 |
| convberturk_light | transformer | ConvBERTurk | light | 0.9366 | 0.9114 | 0.8894 | 0.9372 | 0.9389 | 0.1925 | 0.0184 | 0.0113 |
| electra_light | transformer | Turkish ELECTRA | light | 0.9326 | 0.9056 | 0.8800 | 0.9331 | 0.9406 | 0.1999 | 0.0193 | 0.0116 |
| turkishbertweet_light | transformer | TurkishBERTweet | light | 0.9199 | 0.8867 | 0.8643 | 0.9209 | 0.9260 | 0.2221 | 0.0133 | 0.0126 |
| xlm_r_light | transformer | XLM-RoBERTa base | light | 0.9073 | 0.8738 | 0.8496 | 0.9082 | 0.9113 | 0.2694 | 0.0168 | 0.0137 |
| baseline_sgd_light | linear_baseline | TF-IDF + SGD log-loss | light | 0.8820 | 0.7956 | 0.7969 | 0.8814 | 0.8592 | 0.3342 | 0.0083 | 0.0106 |
| baseline_sgd_raw | linear_baseline | TF-IDF + SGD log-loss | raw | 0.8766 | 0.7798 | 0.7866 | 0.8757 | 0.8452 | 0.3471 | 0.0082 | 0.0104 |
| baseline_sgd_light_masked | linear_baseline | TF-IDF + SGD log-loss | light_masked | 0.8608 | 0.7666 | 0.7642 | 0.8596 | 0.8308 | 0.3830 | 0.0087 | 0.0131 |

Main reading: macro-F1 is the primary metric because the `SEXIST` class is much smaller than the other labels. Accuracy is reported only as supporting context.
- Best transformer: `berturk_light` with macro-F1 `0.8997`.
- Best linear baseline: `baseline_sgd_light` with macro-F1 `0.7969`.
- Absolute macro-F1 lift over the best baseline: `0.1028`.

## 4. Best model class-level behavior
Best run by test macro-F1: `berturk_light` with macro-F1 `0.8997`.
| label | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| OTHER | 0.9740 | 0.9688 | 0.9714 | 5649 |
| PROFANITY | 0.9515 | 0.9459 | 0.9487 | 2738 |
| INSULT | 0.8759 | 0.8602 | 0.8680 | 1617 |
| RACIST | 0.9017 | 0.9325 | 0.9168 | 1525 |
| SEXIST | 0.7378 | 0.8582 | 0.7934 | 141 |

Raw confusion matrix:
![confusion_matrix](results/transformers/berturk_light/confusion_matrix.png)

Normalized confusion matrix:
![confusion_matrix_normalized](results/transformers/berturk_light/confusion_matrix_normalized.png)

Confusion matrix reading:
- `INSULT` -> `OTHER`: `92` örnek, normalized oran `0.06`.
- `SEXIST` -> `PROFANITY`: `8` örnek, normalized oran `0.06`.
- `INSULT` -> `PROFANITY`: `77` örnek, normalized oran `0.05`.
- `SEXIST` -> `INSULT`: `6` örnek, normalized oran `0.04`.
- `INSULT` -> `RACIST`: `45` örnek, normalized oran `0.03`.
- `SEXIST` sınıfında doğru sınıf oranı `0.86`; düşük destek nedeniyle bu sınıf ayrı izlenmeli.

## 5. Calibration and threshold policies
All neural runs use validation-based temperature scaling before test metrics are reported. Threshold policies are selected on validation and evaluated on the held-out test set.
| threshold | accuracy | precision | recall | f1 | flagged_rate | policy |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5800 | 0.9727 | 0.9734 | 0.9736 | 0.9735 | 0.5160 | best_f1 |
| 0.9500 | 0.9569 | 0.9934 | 0.9226 | 0.9567 | 0.4792 | recall_floor_90 |
- `best_f1`: threshold `0.58`, precision `0.9734`, recall `0.9736`, F1 `0.9735`, flagged rate `0.5160`.
- `recall_floor_90`: threshold `0.95`, precision `0.9934`, recall `0.9226`, F1 `0.9567`, flagged rate `0.4792`.

Reliability plot:
![reliability_test](results/transformers/berturk_light/reliability_test.png)

Threshold sweep:
![threshold_sweep_val](results/transformers/berturk_light/threshold_sweep_val.png)

## 6. Confusion matrix appendix
### baseline_sgd_light
![confusion_matrix](results/variant_light/confusion_matrix.png)
### baseline_sgd_light_masked
![confusion_matrix](results/variant_light_masked/confusion_matrix.png)
### baseline_sgd_raw
![confusion_matrix](results/variant_raw/confusion_matrix.png)
### berturk_light
![confusion_matrix](results/transformers/berturk_light/confusion_matrix.png)
### berturk_raw
![confusion_matrix](results/transformers/berturk_raw/confusion_matrix.png)
### convberturk_light
![confusion_matrix](results/transformers/convberturk_light/confusion_matrix.png)
### electra_light
![confusion_matrix](results/transformers/electra_light/confusion_matrix.png)
### turkishbertweet_light
![confusion_matrix](results/transformers/turkishbertweet_light/confusion_matrix.png)
### xlm_r_light
![confusion_matrix](results/transformers/xlm_r_light/confusion_matrix.png)

## 7. Limitations
- Transformer runs use reduced epochs to keep the full model set feasible on a single laptop GPU.
- HPO is intentionally limited to BERTurk and transferred to the other transformer models.
- The dataset is a combined/pseudo-labeled resource, not a clean single-source benchmark.
- `SEXIST` remains the most fragile class because its support is far lower than the other labels.

## 8. Sources
- Dataset: https://huggingface.co/datasets/Overfit-GM/turkish-toxic-language
- BERTurk: https://huggingface.co/dbmdz/bert-base-turkish-cased
- TurkishBERTweet: https://huggingface.co/VRLLab/TurkishBERTweet
- ConvBERTurk: https://huggingface.co/dbmdz/convbert-base-turkish-cased
- XLM-R: https://huggingface.co/FacebookAI/xlm-roberta-base
