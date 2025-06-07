# ART/ARC

This is an implementation of the paper: The ART of Link Prediction with KGEs

AutoRegressive Modelling (Convolution, Transformer)  of the joint distribution: p(S,R,O) \\     

ART (transformer) and ARC (convolution) model the factorized joint distribution p(S)p(R|S)p(O|R,S). 

## Installation 

```bash
conda create -n "art" python=3.10
conda activate art
pip install -r requirements.txt
```



## Train your own model

To see all training options run:

```bash
python -m src.autoregressive_models.train --help
```

Create a config file in the `configs/` folder. Then for example run:

```bash
python -m src.autoregressive_models.train --config configs/fb15k237/example.yaml
```

```bash
python -m experiments.main --experiment link-prediction --config ./configs/fb15k237/example.yaml
```

## Reproduce results of the paper 
For instructions on how to reproduce the results from the paper, please see [reproduce.md](reproduce/reproduce.md).



