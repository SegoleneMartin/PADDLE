# Towards Practical Few-shot Query Sets: Transductive Minimum Description Length Inference


##  Introduction
This repo contains the code for our NeurIPS 2022 submitted paper "Towards Practical Few-shot Query Sets:
Transductive Minimum Description Length Inference". This is a framework that regroups all methods evaluated in our paper. Results provided in the paper can be reproduced with this repo. Code was developed under python 3.8.3 and pytorch 1.4.0.


## 1. Getting started


### 1.1 Quick installation (recommended) (Download datasets and models)
To download datasets tiered and mini and the corresponding pre-trained models (checkpoints), follow instructions 1.1.1 to 1.1.2 of NeurIPS 2020 paper "TIM: Transductive Information Maximization" public implementation (https://github.com/mboudiaf/TIM).
To download the i-Nat dataset for few-shot classification, follow the instructions 2.4 of the ICML 2020 paper "LaplacianShot: Laplacian Regularized Few Shot Learning" public implementation (https://github.com/imtiazziko/LaplacianShot).
#### 1.1.1 Place datasets
Make sure to place the downloaded datasets (data/ folder) at the root of the directory.

#### 1.1.2 Place models
Make sure to place the downloaded pre-trained models (checkpoints/ folder) at the root of the directory.

### 1.2 Manual installation
Follow instruction 1.2 of NeurIPS 2020 paper "TIM: Transductive Information Maximization" public implementation (https://github.com/mboudiaf/TIM) if facing issues with previous steps. Make sure to place data/ and checkpoints/ folders at the root of the directory.

### 2. Requirements
To install requirements:
```bash
conda create --name <env> --file requirements.txt
```
Where \<env> is the name of your environment

## 3. Reproducing the main results

Before anything, activate the environment:
```python
source activate <env>
```

### 3.1 Table 1 and 2 results in paper

To reproduce the results from the paper, from the root of the directory execute this python command.
```python
python3 -m main --base_config <path_to_base_config_file> --method_config <path_to_method_config_file> 
```

The <path_to_base_config_file> follows this hierarchy:
```python
config/<balanced or dirichlet>/base_config/<resnet18 or wideres>/<mini or tiered or inatural/base_config.yaml
```

The <path_to_method_config_file> follows this hierarchy:
```python
config/<balanced or dirichlet>/methods_config/<soft_km or paddle or km_gd_unbiased or alpha_tim or tim or baseline or bdcspn or laplacianshot or pt_map or ici>.yaml
```

For instance, if you want to reproduce the results in the general few-shot setting proposed in the paper, with Keff=5, on mini-Imagenet, using ResNet-18, with PADDLE method go to the root of the directory and execute:
```python
python3 -m src.main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/paddle.yaml
```

# Acknowledgements
Special thanks to the authors of NeurIPS 2020 paper "TIM: Transductive Information Maximization" (TIM) (https://github.com/mboudiaf/TIM) and to the authors of NeurIPS 2021 paper "Realistic evaluation of transductive few-shot learning" (https://github.com/oveilleux/Realistic_Transductive_Few_Shot) for publicly sharing their pre-trained models and their source code from which this repo was inspired from. 


