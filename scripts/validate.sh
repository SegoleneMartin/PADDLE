#!/bin/bash
#SBATCH --mem=0 # Require full memory
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --array=0-?
#SBATCH --time=08:00:00
#SBATCH --account=rrg-ebrahimi

#SBATCH --mail-user=malik.boudiaf.1@etsmtl.net
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL


source ~/.bash_profile
module load python/3.8.2
source ~/ENV/bin/activate

export DATASET_DIR=${SLURM_TMPDIR}/data
mkdir -p ${DATASET_DIR}


# Defining all posssibilites

DATASETS=("mini_imagenet" "tiered_imagenet" "inatural")
METHODS=("alpha_tim" "tim" "laplacianshot" "baseline" "pt_map" "km_unbiased" "ici" "bdcspn")
PARAMETERS_TO_TUNE=("alpha" "loss_weights" "lmd" "iter" "alpha" "alpha" "d" "temp")
VALUES=("[4.0,4.0,4.0] [5.0,5.0,5.0] [6.0,6.0,6.0] [7.0,7.0,7.0] [8.0,8.0,8.0] [9.0,9.0,9.0] [10.0,10.0,10.0]" \
"[1.0,0.1,1.0] [1.0,0.2,1.0] [1.0,0.3,1.0] [1.0,0.4,1.0] [1.0,0.5,1.0] [1.0,0.6,1.0] [1.0,0.7,1.0]" \
"[0.4,0.7] [0.5,0.7] [0.6,0.7] [0.7,0.7] [0.8,0.7] [0.9,0.7] [1.0,0.7]" \
"10 20 30 40 50 60 70" \
"0.004 0.008 0.02 0.06 0.1 0.2 0.5 1 2 3" \
"50 55 60 65 70 75 80" \
"1 3 5 7 9 11 13" \
"2 4 6 8 10 12" \
)

# Defining all required for the current simulation

architecture="resnet18"
dataset_used=${DATASETS[$((SLURM_ARRAY_TASK_ID / 3))]}
method_used=${METHODS[$((SLURM_ARRAY_TASK_ID % 3))]}
parameter_name=${PARAMETERS_TO_TUNE[$((SLURM_ARRAY_TASK_ID % 3))]}
parameter_values=${VALUES[$((SLURM_ARRAY_TASK_ID % 3))]}

tar xf ~/scratch/sego/data/${dataset}.tar.gz -C ${DATASET_DIR}

for method in ${METHODS}; do \
    for value in ${parameter_values}; do \
        python3 -m main --base_config config/dirichlet/base_config/${architecture}/${dataset}/base_config.yaml \
                        --method_config config/dirichlet/methods_config/${method}.yaml \
                        opts ${parameter_name} ${value} n_ways 5 num_classes_test 5 ;\
    done
done
