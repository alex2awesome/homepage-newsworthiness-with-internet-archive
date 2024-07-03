#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=100GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi

# Determine the directory of the current script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

# Run csvtococo.py
python3 "${SCRIPT_DIR}/csvtococo.py" \
    --csv_dir "../small-csv-files" \
    --image_dir "../small-images" \
    --json_file "annotations.json"

# Check if csvtococo.py exited successfully
if [ $? -eq 0 ]; then
    # Run detectronscript.py if csvtococo.py succeeded
    python3 "${SCRIPT_DIR}/detectronscript.py" \
        --dataset_name "HomepageData" \
        --annotations_json "annotations.json" \
        --image_dir "../small-images" \
        --device "cpu"
else
    echo "csvtococo.py failed. Exiting."
    exit 1
fi
