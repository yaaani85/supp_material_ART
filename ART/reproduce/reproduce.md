# To reproduce ART.
Please run:
```bash
./reproduce/train_autoregressive_models.sh
```

# To evaluate the models:



Eval:
```bash
./reproduce/eval_ogblbiokg.sh
./reproduce/eval_fb15k237.sh
./reproduce/eval_wn18rr.sh
```

Statistical significance test:
```bash
./reproduce/statistical_significance_test/stat_transformer_fb15k237.sh
./reproduce/statistical_significance_test/stat_transformer_wn18rr.sh
```




# To reproduce Complex2, and Complex, Complex* and NBF (used in this work).

Please download the following models from gekcs (https://github.com/april-tools/gekcs/): 
### Complex2 

```bash
python -m kbc.experiment --experiment_id MLE --dataset FB15K-237 --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_ll True

python -m kbc.experiment --experiment_id MLE --dataset WN18RR --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 1000 --learning_rate 0.001 --score_ll True

python -m kbc.experiment --experiment_id MLE --dataset ogbl-biokg --model SquaredComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_ll True
```
Then run the experiments with the callbration config e.g.
```bash
python experiments.main.py --experiment link-prediction --config ./configs/fb15k237/complex2.yaml
```

Save the model in "/saved_models/data_dir/complex2/best_filtered_mrr.pt"
### Complex 

```bash
python -m kbc.experiment --experiment_id PLL --dataset FB15K-237 --model ComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True

python -m kbc.experiment --experiment_id PLL --dataset WN18RR --model ComplEx --rank 1000 --optimizer Adam --batch_size 500 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True

python -m kbc.experiment --experiment_id PLL --dataset ogbl-biokg --model ComplEx --rank 1000 --optimizer Adam --batch_size 5000 --learning_rate 0.001 --score_lhs True --score_rel True --score_rhs True
```

Save the model in "/saved_models/data_dir/complex/best_filtered_mrr.pt"
Then run the experiments with the callbration config e.g.
```bash
python experiments.main.py --experiment link-prediction --config ./configs/fb15k237/complex.yaml
```


To reproduce NBF.
Please train the model with the following command in NBF (https://github.com/DeepGraphLearning/NBFNet)


### (optional) Install Torchdrug required to reproduce NBF results
This depends on your cuda version as well. Tested with cuda torch 2.1.0 & CUDA 11.5
However, more cuda versions are possible, please see https://torchdrug.ai/docs/installation.html#from-pip

```bash
pip install torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.0+cu115.html
pip install torchdrug
```


### NBF 

```bash
python script/run.py -c config/knowledge_graph/wn18rr.yaml --gpus [0] 


python script/run.py -c config/knowledge_graph/fb15k237.yaml --gpus [0] 

python script/run.py -c config/knowledge_graph/ogbl-biokg.yaml --gpus [0] 
```
Save the model in "/saved_models/data_dir/nbf/best_filtered_mrr.pt"
Then run the experiments with the callbration config e.g.
```bash
python experiments.main.py --experiment link-prediction --config ./configs/wn18rr/nbf.yaml
```

Then re-name them: complex2, complex, and nbf respecitvely, place them in ./saved_models/data_dir/
## Complex*
To reproduce the calibration-model used in the paper.
Get Complex model (above). 
Run link prediction experiment on validation set with save negatives =20 and save_positives =True
```bash
python external_models/wrappers/callibration.py --dataset FB15K237

```
to train the callibration model. 
Then run the experiments with the callbration config e.g.
```bash
python experiments.main.py --experiment link-prediction --config ./configs/fb15k237/complexC.yaml
```









