#!/bin/bash
#SBATCH --job-name=teacher_search
#SBATCH --output=logs/teacher_%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=gpu,gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --constraint=sm_70|sm_75|sm_80|sm_86
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=gpu013,gpu016,
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7%4
#SBATCH --account=pi_hzamani_umass_edu

# --- Hyperparameter grid (2 x 2 x 2 = 8 combos) ---
LRS=(2e-5 5e-5)
BATCH_SIZES=(32 64)
MAX_LENGTHS=(128 256)

LR=${LRS[$((SLURM_ARRAY_TASK_ID % 2))]}
BATCH_SIZE=${BATCH_SIZES[$(((SLURM_ARRAY_TASK_ID / 2) % 2))]}
MAX_LENGTH=${MAX_LENGTHS[$(((SLURM_ARRAY_TASK_ID / 4) % 2))]}

# --- Conda ---
module load conda/latest
conda activate rag

module load cuda/12.6

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DS=yelp

MODEL_OUTPUT="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/cs682/teacher_${DS}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pt"
TRAIN_OUTPUT="/scratch4/workspace/oyilmazel_umass_edu-rag_collapse/cs682/teacher_${DS}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "Task ${SLURM_ARRAY_TASK_ID}: lr=${LR}, batch_size=${BATCH_SIZE}, max_length=${MAX_LENGTH}"

python cs682/teacher_trainer.py \
    --dataset ${DS} \
    --epochs 3 \
    --learning_rate ${LR} \
    --batch_size ${BATCH_SIZE} \
    --dropout 0.1 \
    --max_length ${MAX_LENGTH} \
    --validation_pct 0.2 \
    --log_every_k 50 \
    --accuracy_every_k 200 \
    --mapped_layer_indices 4 8 \
    --model_save_path "${MODEL_OUTPUT}" \
    --train_save_path "${TRAIN_OUTPUT}"
