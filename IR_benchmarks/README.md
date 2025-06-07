# IR Benchmarks for Link Prediction

This repository contains tools for creating Information Retrieval (IR) evaluation benchmarks for Link Prediction (LP) models. It specifically focuses on generating a subset of LP benchmarks that are suitable for IR metric evaluation.

## Overview

The IR benchmark creation process works by:
1. Taking existing LP benchmarks as input
2. Executing head and tail queries
3. Selecting only the first occurrence of each entity in head or tail position
4. Creating a filtered subset suitable for IR evaluation

## Example Usage with FB15k237

### Prerequisites
- FB15k237 dataset
- Python 3.7+

### Steps to Run

1. First, ensure you have the FB15k237 dataset in your data directory:
```bash
data/
└── FB15k237/
    ├── train.txt
    ├── valid.txt
    └── test.txt

2. Run the benchmark creation script e.g.:
```bash
python main.py --dataset ./FB15k237 
```

This script will generate a new directory in `ir_benchmarks/` with the candidates and their labels.


