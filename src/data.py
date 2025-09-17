# -*- coding: utf-8 -*-
import pandas as pd

LABELS = ["no", "intrinsic", "extrinsic"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for i,l in enumerate(LABELS)}

def build_text(df: pd.DataFrame) -> pd.Series:
    return (
        "CTX: " + df["context"].astype(str) + " \n"
        + "PRM: " + df["prompt"].astype(str) + " \n"
        + "RSP: " + df["response"].astype(str)
    )
