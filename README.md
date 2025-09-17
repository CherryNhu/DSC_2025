# ViHallu Baseline (Giai đoạn 1)

Baseline hợp lệ cho bài toán phân loại hallucination (`no | intrinsic | extrinsic`) từ `context, prompt, response`.
Cung cấp 2 baseline:
1) TF-IDF + LinearSVC (nhanh, không cần GPU)
2) XLM-R base + Classification Head (encoder-only, <=7B)

## Cấu trúc
vihallu-baseline/
├── configs/xlmr_baseline.json
├── outputs/
├── scripts/run_tfidf.sh
├── scripts/run_xlmr.sh
├── src/data.py
├── src/utils.py
├── src/models/tfidf_svm.py
├── src/models/xlmr_baseline.py
└── requirements.txt

## Dữ liệu
- /mnt/data/vihallu-train.csv (id, context, prompt, response, label)
- /mnt/data/vihallu-public-test.csv (id, context, prompt, response)

## Cài & chạy
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# TF-IDF
bash scripts/run_tfidf.sh
# XLM-R
bash scripts/run_xlmr.sh
