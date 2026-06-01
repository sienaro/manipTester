"""
Step 4: Error analysis — confidence distribution plot and threshold sweep.

Requires: saved_model_oversampled/ from step 3, speech_labels_final.csv
Run: python 4_error_analysis.py
Saves: confidence_dist.png
"""

import csv, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(texts, model, tokenizer, threshold=0.5):
    model.eval()
    all_preds, all_probs = [], []
    for i in range(0, len(texts), 16):
        enc = tokenizer(texts[i:i+16], truncation=True, padding=True,
                        max_length=256, return_tensors='pt')
        with torch.no_grad():
            probs = torch.softmax(
                model(input_ids=enc['input_ids'].to(DEVICE),
                      attention_mask=enc['attention_mask'].to(DEVICE)).logits,
                dim=1)[:,1].cpu().numpy()
        all_probs.extend(probs); all_preds.extend((probs >= threshold).astype(int))
    return np.array(all_preds), np.array(all_probs)


if __name__ == "__main__":
    # Load speech labels (same split as step 3)
    sp_texts, sp_labels, sp_meta = [], [], []
    with open("speech_labels_final.csv", 'r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sp_texts.append(r['text'])
            sp_labels.append(int(r['label']))
            sp_meta.append((r.get('president','?'), r.get('title','?')))

    _, X_sp_test, _, y_sp_test = train_test_split(
        sp_texts, sp_labels, test_size=0.2, random_state=SEED, stratify=sp_labels)

    # Load oversampled model
    tokenizer = AutoTokenizer.from_pretrained("./saved_model_oversampled")
    model     = AutoModelForSequenceClassification.from_pretrained(
        "./saved_model_oversampled").to(DEVICE)

    preds, probs = run_inference(X_sp_test, model, tokenizer)

    # ── Confidence distribution plot ──
    manip    = [probs[i] for i in range(len(y_sp_test)) if y_sp_test[i] == 1]
    nonmanip = [probs[i] for i in range(len(y_sp_test)) if y_sp_test[i] == 0]

    plt.figure(figsize=(8, 4))
    plt.hist(nonmanip, bins=20, alpha=0.6, label='Non-manipulative', color='steelblue')
    plt.hist(manip,    bins=20, alpha=0.6, label='Manipulative',     color='tomato')
    plt.axvline(0.5, color='black', linestyle='--', label='Threshold=0.5')
    plt.xlabel('Model confidence (P(manipulative))')
    plt.ylabel('Count')
    plt.title('Confidence Distribution on Politician Speeches')
    plt.legend()
    plt.tight_layout()
    plt.savefig('confidence_dist.png', dpi=150)
    print("Saved confidence_dist.png")

    # ── Threshold sweep ──
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Macro F1':>10}")
    print("-" * 45)
    for thresh in [0.50, 0.40, 0.35, 0.25, 0.15]:
        p_ = (probs >= thresh).astype(int)
        p  = precision_score(y_sp_test, p_, pos_label=1, zero_division=0)
        r  = recall_score(y_sp_test,    p_, pos_label=1, zero_division=0)
        f  = f1_score(y_sp_test,        p_, average='macro', zero_division=0)
        print(f"{thresh:>10.2f} {p:>10.3f} {r:>10.3f} {f:>10.3f}")

    # ── False positives ──
    # Run on full speech dataset for false positive taxonomy
    all_preds, all_probs = run_inference(sp_texts, model, tokenizer)
    fp = [(all_probs[i], sp_texts[i], sp_meta[i])
          for i in range(len(sp_labels)) if sp_labels[i]==0 and all_preds[i]==1]
    fp.sort(reverse=True)
    print(f"\nTotal false positives: {len(fp)}")
    for prob, text, (pres, title) in fp[:10]:
        print(f"[{prob:.3f}] {pres} — {title}\n  \"{text[:150]}\"")
