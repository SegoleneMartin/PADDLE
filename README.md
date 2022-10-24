# Towards Practical Few-shot Query Sets: Transductive Minimum Description Length Inference


##  Introduction
This repo contains the code for our NeurIPS 2022 paper "Towards Practical Few-shot Query Sets:
Transductive Minimum Description Length Inference", available at INSERT LINK. 

It includes our two main contributions:

- the generation of realistic few-shot tasks with a very large number $K$ of ways, and an adjustable number $K_{\mathrm{eff}}$ of classes effectively present in the query set, 

- our classifier PADDLE (see Figure below) as well as all the other methods evaluated in our paper.

Results provided in the paper can be reproduced with this repo. Code was developed under python 3.8 and pytorch 1.12.1.

<img src="framework.png" scale=1/>

## 1. Getting started

### 1.1. Requirements
Create a new conda environment using the .yml file provided.
```bash
conda env create -f paddle_env.yml
```

### 1.2 Download datasets and models
Our framework was developped for the datasets mini-imagenet, tiered-imagenet and iNatural. We used pre-trained models. 

The downloaded datasets should be placed in the folder data/ the following way:

    .
    ├── ...
    ├── data                    
    │   ├── mini_imagenet          
    │   ├── tiered_imagenet        
    │   └── inatural               
    └── ...

The downloaded models should be placed in the folder checkpoints/ the following way:

    .
    ├── ...
    ├── checkpoints                    
    │   ├── mini          
    │   ├── tiered        
    │   └── inatural               
    └── ...

#### 1.2.1 Mini-imagenet and tiered-imagenet

We follow instructions 1.1.1 to 1.1.2 of NeurIPS 2020 paper "TIM: Transductive Information Maximization" public implementation (https://github.com/mboudiaf/TIM).

For the datasets, please download the zip file at https://drive.google.com/drive/folders/163HGKZTvfcxsY96uIF6ILK_6ZmlULf_j?usp=sharing, and unzip it into the data/ folder.

For the corresponding pre-trained models, please download the zip file at https://drive.google.com/file/d/15MFsig6pjXO7vZdo-1znJoXtHv4NY-AF/view?usp=sharing, and unzip it at the root.

#### 1.2.2 Inatural

To download the iNatural dataset for few-shot classification, we follow the instructions 2.4 of the ICML 2020 paper "LaplacianShot: Laplacian Regularized Few Shot Learning" public implementation (https://github.com/imtiazziko/LaplacianShot).


## 3. Running the code and reproducing the main results

Before anything, activate the environment:
```python
conda activate paddle
```

### 3.1 Table 1 and 2 results in paper

If you want to reproduce the results in the realistic few-shot setting proposed in the paper, on mini-Imagenet, for $20$-shots tasks with $K_{\mathrm{eff}} = 5$, using a ResNet-18 and PADDLE classifier, go to the root of the directory and execute:
```python
python3 -m src.main --opts dataset mini shots [20] k_eff 5 arch resnet18 method paddle 
```
You might also want to directly modify the options in the following config file: config/main_config.yalm


# Acknowledgements
Special thanks to the authors of NeurIPS 2020 paper "TIM: Transductive Information Maximization" (TIM) (https://github.com/mboudiaf/TIM) and to the authors of NeurIPS 2021 paper "Realistic evaluation of transductive few-shot learning" (https://github.com/oveilleux/Realistic_Transductive_Few_Shot) for publicly sharing their pre-trained models and their source code from which this repo was inspired from. 


