# -*- coding: utf-8 -*-

# Cat chuoi uu tien giu RSP -> Han che mat thong tin trong response (thuong la tin hieu manh nhat)

import pandas as pd 
from transformers import AutoTokenizer

LABELS = ["no", "intrinsic", "extrinsic"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for i,l in enumerate(LABELS)}

def _concat(context: str, prompt: str, response: str) -> str:
    return f"CTX: {context}\nPRM: {prompt}\nRSP: {response}"

# Dung khi khong can cat  
def build_text(df: pd.DataFrame) -> pd.Series:
    return (
        "CTX: " + df["context"].astype(str) + " \n"
        + "PRM: " + df["prompt"].astype(str) + " \n"
        + "RSP: " + df["response"].astype(str)
    )

""" 
Cat chuoi theo nguyen tac:
- Luon giu full RSP
- Giu PRM
- Cat CTX neu tong > max_length
"""
def build_text_truncate(df: pd.DataFrame, tokenizer_name: str, max_length: int) -> pd.Series:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    outs = []
    for _, row in df.iterrows():
        ctx, prm, rsp = str(row["context"]), str(row["prompt"]), str(row["response"])
        base = f"PRM: {prm}\nRSP: {rsp}"
        
        # Thu ghep day du 
        full = _concat(ctx, prm, rsp)
        if len(tok(full, add_special_tokens=True)["input_ids"]) <= max_length:
            outs.append(full); continue
        
        # Neu qua dai thi cat CTX
        # Cat “tho” theo so token gan dung: giam theo tung phan 10%
        ctx_tokens = tok(ctx, add_special_tokens=False)["input_ids"]
        keep = len(ctx_tokens)
        while keep > 0:
            trial = _concat(tok.decode(ctx_tokens[:keep], skip_special_tokens=True), prm, rsp)
            if len(tok(trial, add_special_tokens=True)["input_ids"]) <= max_length:
                outs.append(trial); break
            keep = int(keep * 0.9)
        if keep <= 0:
            outs.append(base)  # cung lam chi con PRM+RSP
    return pd.Series(outs)