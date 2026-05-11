# Project 4 Experiment Report

## Dataset audit
- Raw rows: `77800`
- Clean rows after conflict and exact duplicate removal: `77800`
- Duplicate text rows: `0`
- Conflicting text count: `0`
- Class distribution: `{'OTHER': 37663, 'PROFANITY': 18252, 'INSULT': 10777, 'RACIST': 10163, 'SEXIST': 945}`

## Experiment summary
| variant | model | temperature | test_accuracy | test_macro_f1 | test_macro_pr_auc | test_top_label_ece | test_classwise_ece |
| --- | --- | --- | --- | --- | --- | --- | --- |
| light | sgd | 1.0500 | 0.8820 | 0.7969 | 0.8592 | 0.0083 | 0.0106 |
| raw | sgd | 1.0500 | 0.8766 | 0.7866 | 0.8452 | 0.0082 | 0.0104 |
| light_masked | sgd | 1.1500 | 0.8608 | 0.7642 | 0.8308 | 0.0087 | 0.0131 |

## Best linear run
- Best preprocessing variant: `light`
- Test macro-F1: `0.7969`
- Test accuracy: `0.8820`
- Test macro PR-AUC: `0.8592`
- Top-label ECE after temperature scaling: `0.0083`

## Toxic / other policies selected on validation
| policy | threshold |
| --- | --- |
| best_f1 | 0.44 |
| recall_floor_90 | 0.64 |

## Generated artifacts
- `audit_summary.json`
- `experiment_summary.csv`
- `best_per_class_report.csv`
- `best_confusion_matrix.csv`
- `threshold_sweep_val.csv`
- `threshold_policies_test.csv`
- `plots/`
