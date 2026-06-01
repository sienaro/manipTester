"""
Step 2: Train BERT on MentalManip and evaluate on in-domain test set
       and hand-labeled politician speeches (OOD).

Run: python 2_train_bert.py
Saves model to: ./saved_model/
Prints: majority baseline, in-domain BERT results, OOD speech results
"""

import csv, random, numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ──────────────────────────────────────────────────────────────

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


def load_speech_labels(filepath):
    rows, texts, labels, meta = [], [], [], []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        texts.append(r['text'])
        labels.append(int(r['label']))
        meta.append((r.get('president', '?'), r.get('title', '?')))
    return texts, labels, meta


# ── Dataset ───────────────────────────────────────────────────────────────────

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


# ── Evaluate ──────────────────────────────────────────────────────────────────

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
        all_probs.extend(probs)
        all_preds.extend((probs >= threshold).astype(int))
    return np.array(all_preds), np.array(all_probs)


# ── Train ─────────────────────────────────────────────────────────────────────

def train(X_train, X_val, X_test, y_train, y_val, y_test,
          tokenizer, model, epochs=3, lr=2e-5, batch_size=16):

    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(cw, dtype=torch.float).to(DEVICE))

    train_loader = DataLoader(DialogueDataset(X_train, y_train, tokenizer),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(DialogueDataset(X_val, y_val, tokenizer),
                              batch_size=batch_size)
    test_loader  = DataLoader(DialogueDataset(X_test, y_test, tokenizer),
                              batch_size=batch_size)

    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=len(train_loader)*epochs//10,
        num_training_steps=len(train_loader)*epochs)

    best_f1, best_state = 0, None
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(train_loader):.4f}  val_f1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print("  ↑ new best")

    model.load_state_dict(best_state)
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            p = torch.softmax(
                model(input_ids=batch['input_ids'].to(DEVICE),
                      attention_mask=batch['attention_mask'].to(DEVICE)).logits,
                dim=1)[:,1].cpu().numpy()
            probs.extend(p); preds.extend((p >= 0.5).astype(int))

    return model, np.array(preds), np.array(probs)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load MentalManip
    data = load_mentalmanip("mentalmanip_maj.csv")
    texts  = [t for t, _ in data]
    labels = [l for _, l in data]
    print(f"MentalManip: {len(data)} dialogues | class dist: {Counter(labels)}")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.125, random_state=SEED, stratify=y_tmp)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Majority baseline
    majority = Counter(y_train).most_common(1)[0][0]
    evaluate(y_test, [majority]*len(y_test), name="Majority Class Baseline")

    # BERT
    print(f"\nDevice: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model     = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2).to(DEVICE)

    print("\n=== Training BERT on MentalManip ===")
    model, preds, probs = train(X_train, X_val, X_test, y_train, y_val, y_test,
                                tokenizer, model)
    evaluate(y_test, preds, probs, name="Fine-tuned BERT (in-domain)")

    # Save model
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("Model saved to ./saved_model")

    # OOD: politician speeches
    sp_texts, sp_labels, sp_meta = load_speech_labels("speech_labels_final.csv")
    sp_preds, sp_probs = run_inference(sp_texts, model, tokenizer)
    evaluate(sp_labels, sp_preds, sp_probs, name="BERT on Politician Speeches (OOD)")

    # False positives and negatives
    fp = [(sp_probs[i], sp_texts[i], sp_meta[i])
          for i in range(len(sp_labels)) if sp_labels[i]==0 and sp_preds[i]==1]
    fn = [(sp_probs[i], sp_texts[i], sp_meta[i])
          for i in range(len(sp_labels)) if sp_labels[i]==1 and sp_preds[i]==0]
    fp.sort(reverse=True); fn.sort()

    print("\n── Top False Positives ──")
    for prob, text, (pres, title) in fp[:5]:
        print(f"[{prob:.3f}] {pres} — {title}\n  \"{text[:150]}\"")

    print("\n── Top False Negatives ──")
    for prob, text, (pres, title) in fn[:5]:
        print(f"[{prob:.3f}] {pres} — {title}\n  \"{text[:150]}\"")
