#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=400G
#SBATCH --partition=sched_mit_psfc_gpu_r8


source /home/spangher/.bashrc
conda activate alex

python run_prompts.py \
  --input_data_file prompts_to_clean.csv.gz \
  --id_col index \
  --prompt_col prompt \
  --output_file  cleaned_prompts.jsonl \

