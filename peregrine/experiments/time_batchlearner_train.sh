#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --job-name=pong_dqn_replay_buffer
#SBATCH --mem=1GB

module purge

cd thesis
source peregrine/activate_thesis_env.sh

bash peregrine/mongo_singularity.sh
# give some time to container to startup
sleep 10

cd dopamine
python -m thesis.experiments.peregrine_time_train_iter
