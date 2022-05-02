#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=6GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --profile=task

# NOTE last directive is for job profiling and viz on Grafana

# NOTE cl switches:
# -J, --job-name
# -o, --output

module purge

cd thesis
source peregrine/activate_thesis_env.sh

bash peregrine/start_mongo.sh
# give some time to container to startup
sleep 5

cd dopamine
python -m $1

# results
# -classic control (DQVMax, CartPole):
# ~27/30s on my pc and pg interactive gpu node
# ~0.027 on pg gpu job, x1000 speedup ?!
