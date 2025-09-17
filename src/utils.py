# -*- coding: utf-8 -*-
from sklearn.metrics import f1_score, accuracy_score

def macro_f1_acc(y_true, y_pred):
    macro = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    return macro, acc
