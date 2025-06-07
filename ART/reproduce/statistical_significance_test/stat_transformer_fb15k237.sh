#!/bin/bash
set -e

declare -a seeds=("12" "17" "13" "2" "3" "5" "9" "2" "22" "8")

for seed in "${seeds[@]}"
do
    python3 src/autoregressive_models/train.py --config ./configs/fb15k237/transformer.yaml --seed "$seed"

    python3 experiments/main.py --experiment link-prediction --config ./configs/fb15k237/transformer.yaml --seed "$seed" >> ./reproduce/statistical_significance_test/fb15k237_transformer.txt

done
