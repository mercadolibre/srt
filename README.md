# Scaling Sequential Recommendation Models with Transformers

This repository contains the code for the paper "Scaling Sequential Recommendation Models with Transformers" 
published at The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval 
(SIGIR 2024)


## Contents

1. [Installing](#installing)
2. [Scaling Datasets](#scaling-datasets)
3. [Pretrain a model](#pretrain-a-model)
4. [Use MLFLOW server to check metrics](#use-mlflow-server-to-check-metrics) 
5. [Fine-tune a model](#fine-tune-a-model)
6. [Recbole fork](#recbole-fork)
7. [Contact](#contact)
8. [Citation](#citation)

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
python scripts/pre_train.py -c confs/pre_train/amazon-100K -o pretrain_checkpoint.pth
```

The model will be written on the file `pretrain_checkpoint.pth`

## Use MLFLOW server to check metrics 

Either while pre-training, fine-tuning or after training a model you can use MLFLOW to check the metrics.
To start the MLFLOW server run

```commandline
mlflow server --backend-store-uri sqlite:///mlruns.sqlite
```

Make sure to have the `mlruns.sqlite` file on the same folder where you run the command.

## Fine-tune a model

The fine-tuning script is a little more complex. To train a model from scratch run 

```commandline
python scripts/fine_tune.py -c confs/fine_tune/from_scratch.json -s amazon-beauty
```

The dataset can be either `amazon-beauty` or `amazon-sports` which are created by the ETLs. 
Also it can be any amazon dataset available on [Recbole](https://recbole.io/)

To fine tune a pre-trained model run

```commandline
python scripts/fine_tune.py -c <config_file> -s amazon-beauty -l $PRETRAIN_DATASET -i $CHECKPOINT_FNAME 
-p $PRETRAIN_CONFIG -o fine_tuned_checkpoint.pth
```

For example
```commandline
python scripts/fine_tune.py -c confs/fine_tune/progressive.json -s amazon-beauty -l amazon-100K -i test.pth -p confs/pre_train/config_dict_100K.json -o fine_tuned_checkpoint.pth
```
This fine tunes a model pretrained on `$PRETRAIN_DATASET` (e.g. amazon-1M) with the checkpoint `$CHECKPOINT_FNAME` 
and the configuration file `$PRETRAIN_CONFIG` (e.g. `confs/config_dict_1M.json`)

## Recbole fork
There were some limitations on Recbole so we decide to fork it. We forked from the commit [`321bff8fc`](https://github.com/RUCAIBox/RecBole/commit/321bff8fc169415c908cd3e722d681b89bee5187) 
To summarize the changes are

1. Better implementation of negative sampling (to use only one forward pass instead of one for each negative sample)
2. Evaluate NDCG both on train, valid and test to check for overfitting
3. Log metrics into mlflow, which is more convenient to analyze the results
4. Compute total parameters and non-embedding parameters
5. Implement one cycle learning rate scheduler

## Contact

For comments, questions, or suggestions please contact [me](mailto:pablo.rzivic@mercadolibre.com). You can also reach me
on twitter at [@ideasrapidas](https://twitter.com/ideasrapidas)


## Notice

This project uses several dependencies as seen in requirements files. 
The developers of this project are not responsible for any potential problems or vulnerabilities that may arise when using this code.

## License
This project is licensed under the terms of the MIT License. See LICENSE.

## Citation
TODO

