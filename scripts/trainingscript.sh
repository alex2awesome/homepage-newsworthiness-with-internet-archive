#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=100GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

# Load necessary modules (adjust as per your environment)
module load cuda/12.1

# Determine the directory of the current script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# Install Python packages from requirements.txt using pip
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Install required packages using Conda
conda install -y pytorch torchvision torchaudio cudatoolkit=12.1 -c pytorch -c nvidia

# Install Detectron2 from GitHub
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Activate your Conda environment if needed (adjust as per your setup)
# conda activate <your_environment_name>

# Run csvtococo.py
python3 "${SCRIPT_DIR}/multiprocesscsvtococo.py" \
    --csv_dir "../../data/bounding-box-results" \
    --image_dir "../../data/bounding-box-results" \
    --json_file "annotations.json"

# Check if csvtococo.py exited successfully
if [ $? -eq 0 ]; then
    # Run detectronscript.py if csvtococo.py succeeded
    python3 "${SCRIPT_DIR}/detectronscript.py" \
        --dataset_name "HomepageData" \
        --annotations_json "annotations.json" \
        --image_dir "../../data/bounding-box-results" \
        --device "cuda" \
        --num_gpus 4
else
    echo "csvtococo.py failed. Exiting."
    exit 1
fi
