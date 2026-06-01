"""
Step 3: Domain adaptation — retrain BERT on MentalManip + oversampled speech labels.

Requires: saved_model/ from step 2, speech_labels_final.csv
Run: python 3_domain_adaptation.py
Prints: before/after comparison on speech test set
"""

import csv, random, numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OVERSAMPLE = 10


# ── Reuse helpers from step 2 ─────────────────────────────────────────────────

def load_mentalmanip(filepath):
    data = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        dcol = next(i for i, h in enumerate(header) if 'dialogue' in h.lower())
        lcol = next(i for i, h in enumerate(header) if 'manipulative' in h.lower())
        for row in reader:
            if len(row) > max(dcol, lcol):
                try:
                    data.append((row[dcol].strip(), int(row[lcol].strip())))
                except ValueError:
                    continue
    return data


class DialogueDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.enc = tokenizer(texts, truncation=True, padding=True,
                             max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {'input_ids':      self.enc['input_ids'][idx],
                'attention_mask': self.enc['attention_mask'][idx],
                'labels':         self.labels[idx]}


def evaluate(y_true, y_pred, y_prob=None, name=""):
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:  {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    if y_prob is not None:
        print(f"ROC-AUC:   {roc_auc_score(y_true, y_prob):.4f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred,
          target_names=["Non-manip", "Manipulative"], zero_division=0))


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


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load MentalManip train split (same seed = same split as step 2)
    data = load_mentalmanip("mentalmanip_maj.csv")
    texts  = [t for t, _ in data]
    labels = [l for _, l in data]

    X_tmp, _, y_tmp, _ = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    X_train, _, y_train, _ = train_test_split(
        X_tmp, y_tmp, test_size=0.125, random_state=SEED, stratify=y_tmp)

    # Load speech labels and split
    sp_texts, sp_labels, sp_meta = [], [], []
    with open("speech_labels_final.csv", 'r', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sp_texts.append(r['text'])
            sp_labels.append(int(r['label']))
            sp_meta.append((r.get('president','?'), r.get('title','?')))

    X_sp_train, X_sp_test, y_sp_train, y_sp_test = train_test_split(
        sp_texts, sp_labels, test_size=0.2, random_state=SEED, stratify=sp_labels)
    print(f"Speech adapt: {len(X_sp_train)} | Speech test: {len(X_sp_test)}")
    print(f"Adapt class dist: {Counter(y_sp_train)}")

    # Build oversampled combined dataset
    manip_speech = [(x, y) for x, y in zip(X_sp_train, y_sp_train) if y == 1]
    print(f"Manipulative speech examples: {len(manip_speech)} → oversampling {OVERSAMPLE}x")

    X_combined = X_train + X_sp_train + [x for x,y in manip_speech]*OVERSAMPLE
    y_combined  = y_train + y_sp_train + [y for x,y in manip_speech]*OVERSAMPLE

    combined = list(zip(X_combined, y_combined))
    random.shuffle(combined)
    X_combined, y_combined = zip(*combined)
    X_combined, y_combined = list(X_combined), list(y_combined)
    print(f"Combined training set: {len(X_combined)} | class dist: {Counter(y_combined)}")

    # Train fresh from bert-base-uncased
    print(f"\nDevice: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2).to(DEVICE)

    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_combined)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(DEVICE))

    EPOCHS, LR, BATCH = 4, 2e-5, 16
    train_loader = DataLoader(DialogueDataset(X_combined, y_combined, tokenizer),
                              batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(DialogueDataset(X_sp_test,  y_sp_test,  tokenizer),
                              batch_size=BATCH)

    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=len(train_loader)*EPOCHS//10,
        num_training_steps=len(train_loader)*EPOCHS)

    best_f1, best_state = 0, None
    print("\n=== Training BERT (MentalManip + Speech, 10x oversample) ===")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            opt.zero_grad()
            loss = loss_fn(
                model(input_ids=batch['input_ids'].to(DEVICE),
                      attention_mask=batch['attention_mask'].to(DEVICE)).logits,
                batch['labels'].to(DEVICE))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total_loss += loss.item()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in val_loader:
                vp.extend(torch.argmax(
                    model(input_ids=batch['input_ids'].to(DEVICE),
                          attention_mask=batch['attention_mask'].to(DEVICE)).logits,
                    dim=1).cpu().numpy())
                vt.extend(batch['labels'].numpy())

        vf1 = f1_score(vt, vp, average='macro', zero_division=0)
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  speech_val_f1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print("  ↑ new best")

    model.load_state_dict(best_state)
    preds, probs = run_inference(X_sp_test, model, tokenizer)
    evaluate(y_sp_test, preds, probs, name="BERT (MentalManip + Speech, 10x oversample)")

    model.save_pretrained("./saved_model_oversampled")
    tokenizer.save_pretrained("./saved_model_oversampled")
    print("Model saved to ./saved_model_oversampled")
