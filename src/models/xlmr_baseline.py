# -*- coding: utf-8 -*-
import argparse, json, numpy as np, pandas as pd, torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from ..data import build_text, label2id, id2label

def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score, accuracy_score
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    macro = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"f1_macro": macro, "accuracy": acc}

def train_xlmr(train_path: str, cfg_path: str, out_dir: str):
    cfg = json.load(open(cfg_path, "r", encoding="utf-8"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(train_path)
    df["text"] = build_text(df)
    df["labels"] = df["label"].map(label2id).astype(int)
    tr_df, val_df = train_test_split(df[["text","labels"]], test_size=0.15, random_state=cfg["seed"], stratify=df["labels"])
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=cfg["max_length"])
    ds = {
        "train": Dataset.from_pandas(tr_df.reset_index(drop=True)).map(tok_fn, batched=True, remove_columns=["text"]),
        "val": Dataset.from_pandas(val_df.reset_index(drop=True)).map(tok_fn, batched=True, remove_columns=["text"]),
    }
    collator = DataCollatorWithPadding(tokenizer=tok)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=3, id2label=id2label, label2id={v:k for k,v in id2label.items()}
    )
    args = TrainingArguments(
        output_dir=str(out/"xlmr-baseline"),
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        num_train_epochs=cfg["epochs"],
        weight_decay=cfg["weight_decay"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=cfg["seed"],
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        tokenizer=tok, data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    best_dir = out/"xlmr-baseline"/"best"
    trainer.save_model(str(best_dir)); tok.save_pretrained(str(best_dir))
    print(f"Saved best checkpoint to {best_dir}")
    return str(best_dir)

def predict_xlmr(model_dir: str, test_path: str, out_csv: str, max_length: int):
    df = pd.read_csv(test_path)
    df["text"] = build_text(df)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); model.eval()
    preds = []
    for i in range(0, len(df), 32):
        batch = df["text"].iloc[i:i+32].tolist()
        enc = tok(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
    inv = {v:k for k,v in label2id.items()}
    labels = [inv[p] for p in preds]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "label": labels}).to_csv(out_csv, index=False)
    print(f"Saved submission to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","predict"], required=True)
    ap.add_argument("--train_path", type=str, default="data/vihallu-train.csv")
    ap.add_argument("--test_path", type=str, default="data/vihallu-public-test.csv")
    ap.add_argument("--cfg", type=str, default="configs/xlmr_baseline.json")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--model_dir", type=str, default="outputs/xlmr-baseline/best")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--out_csv", type=str, default="outputs/submission_xlmr.csv")
    args = ap.parse_args()
    if args.mode == "train":
        train_xlmr(args.train_path, args.cfg, args.out_dir)
    else:
        predict_xlmr(args.model_dir, args.test_path, args.out_csv, args.max_length)
