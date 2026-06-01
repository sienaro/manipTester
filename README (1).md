# CS221 Final Project: Psychological Manipulation Detection in Political Speech

Detects psychological manipulation in text using fine-tuned BERT, trained on the
MentalManip dataset and evaluated on hand-labeled US presidential speeches.

## Files

| File | Description |
|---|---|
| `1_download_data.py` | Downloads MentalManip + Miller Center speeches |
| `2_train_bert.py` | Trains BERT on MentalManip, evaluates in-domain + OOD |
| `3_domain_adaptation.py` | Retrains with oversampled speech labels |
| `4_error_analysis.py` | Confidence distribution plot + threshold sweep |
| `speech_labels_final.csv` | 217 hand-labeled presidential speech segments |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
python 1_download_data.py
```

## Reproduce Results

```bash
python 2_train_bert.py        # majority baseline + BERT in-domain + OOD
python 3_domain_adaptation.py # domain adaptation with oversampling
python 4_error_analysis.py    # confidence plot + threshold sweep
```

Requires GPU (T4 recommended). Steps 2 and 3 take ~10 min each on T4.

## Results

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Majority Class Baseline | 0.705 | 0.414 | N/A |
| Fine-tuned BERT (in-domain) | 0.690 | 0.639 | 0.744 |
| BERT on Politician Speeches (OOD) | 0.894 | 0.546 | 0.724 |
| BERT + Speeches (10x oversample) | 0.864 | 0.588 | 0.810 |

## References

1. Wang et al. (2024). MentalManip. ACL 2024.
2. Krak et al. (2024). Political propaganda detection. Problems in Programming.
