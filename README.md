# Scaling Sequential Recommendation Models with Transformers

This repository contains the code for the paper "Scaling Sequential Recommendation Models with Transformers" 
published at The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval 
(SIGIR 2024)

## Installing

To install the required packages you need to use either `requirements_cpu.txt` if you are installing on a CPU 
or `requirements_gpu.txt` if you are installing on a GPU. Using a computer with many CPU cores will speed up
ETL since it is parallelized via multicore. For pre-training and fine-tuning GPU is required.

- CPU: `pip install -r requirements_cpu.txt`
- GPU: `pip install -r requirements_gpu.txt`

## Scaling Datasets

To compute the scaled datasets from the [Amazon Product Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) run

```commandline
python scripts/scaling_datasets.py --step all
```

## Pretrain a model

The pre-training script reads a configuration file and trains a model on the specified dataset. 
There are three configuration files on the `confs/pre_train` folder to pre-train on 100K samples, 1M samples, and 10M samples.

```commandline
python scripts/pre_train.py -c <config_file>
```

The model will be written on **<TODO>**

## Fine-tune a model

The fine-tuning script is a little more complex. To train a model from scratch run 

```commandline
python scripts/fine_tune.py -c <config_file> -s amazon-beauty
```
The dataset can be either `amazon-beauty` or `amazon-sports` which are created by the ETLs. 
Also it can be any amazon dataset available on [Recbole](https://recbole.io/)


To fine tune a pre-trained model run

```commandline
python scripts/fine_tune.py -c <config_file> -s amazon-beauty -l $PRETRAIN_DATASET -i $CHECKPOINT_FNAME -p $PRETRAIN_CONFIG
```

This fine tunes a model pretrained on `$PRETRAIN_DATASET` (e.g. amazon-1M) with the checkpoint `$CHECKPOINT_FNAME` 
(writen on the `saved` folder by `recbole`) and the configuration file `$PRETRAIN_CONFIG` (e.g. `confs/config_dict_1M.json`)

## Recbole fork
There were some limitations on Recbole so we decide to fork it. To summarize the changes are

1. Better implementation of negative sampling (to use only one forward pass instead of one for each negative sample)
2. Evaluate NDCG both on train, valid and test to check for overfitting
3. Log metrics into mlflow, which is more convenient to then analyze the results
4. **TODO are there more?**
5. 
## Citation
