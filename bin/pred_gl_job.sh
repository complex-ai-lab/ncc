#!/bin/bash

#SBATCH --partition=spgpu
#SBATCH --time=00-24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=47GB
#SBATCH --account=alrodri0

# set up job
module load python/3.9.12 cuda
# pushd /home/liruipu/End2EndCP/
pushd /home/liruipu/e2ecp/
source ../CDC-FluSight-2023/env/bin/activate

# run online training
cd src/forecaster
python online_training.py --input $1
