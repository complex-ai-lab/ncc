#!/bin/bash

#SBATCH --partition=gpu_mig40,spgpu
#SBATCH --time=02-00:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=47GB
#SBATCH --account=alrodri0
#SBATCH --mail-user=liruipu@umich.edu
#SBATCH --mail-type=END


# set up job
module load python/3.9.12 cuda
# pushd /home/liruipu/End2EndCP/
pushd /home/liruipu/e2ecp/
source ../CDC-FluSight-2023/env/bin/activate

# run online training
cd src/forecaster
python e2ecp.py --input $1 -t -g
python e2ecp.py --input $1 -g
