#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=6GB
#SBATCH --job-name=time_dqvmax_train_iter_cc
#SBATCH --output=time_dqvmax_train_iter_cc-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --profile=task
# NOTE last directive is for job profiling and viz on Grafana

module purge

cd thesis
source peregrine/activate_thesis_env.sh

bash peregrine/start_mongo.sh
# give some time to container to startup
sleep 10

cd dopamine
python -m thesis.experiments.pg_time_train_iter_cc

# results:
# ~27/30s on my pc and pg interactive gpu node
# ~0.027 on pg gpu job, x1000 speedup ?!
