#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --job-name=pong_dqn_replay_buffer
#SBATCH --mem=1GB

module purge
cd thesis
source peregrine/load_modules.sh
source $(poetry env info --path)/bin/activate
gsutil -m cp -R gs://atari-replay-datasets/dqn/Pong /data/$USER
