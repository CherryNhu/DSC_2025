#!/usr/bin/env bash
set -e
python -m src.models.tfidf_svm --train_path data/vihallu-train.csv --test_path data/vihallu-public-test.csv --out_dir outputs
