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

DATASETS=("mini_imagenet" "tiered_imagenet" "meta_inat")
METHODS=("ici" "pt_map" "tim_gd")
PARAMETERS_TO_TUNE=("d" "")
VALUES=("0.1 0.5 1.0" )

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
                        --method_config config/dirichlet/methods_config/${method}.yaml ;\
                        opts ${parameter_name} ${value}
done
