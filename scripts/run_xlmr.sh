#!/usr/bin/env bash
set -e
python -m src.models.xlmr_baseline --mode train --train_path data/vihallu-train.csv --cfg configs/xlmr_baseline.json --out_dir outputs
python -m src.models.xlmr_baseline --mode predict --test_path data/vihallu-public-test.csv --model_dir outputs/xlmr-baseline/best --out_csv outputs/submission_xlmr.csv --max_length 512
