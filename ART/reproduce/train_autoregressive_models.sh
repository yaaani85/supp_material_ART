#!/bin/bash

set -e

python3 src/autoregressive_models/train.py --config ./configs/fb15k237/transformer.yaml
python3 src/autoregressive_models/train.py --config ./configs/wn18rr/transformer.yaml
python3 src/autoregressive_models/train.py --config ./configs/ogblbiokg/transformer.yaml  
