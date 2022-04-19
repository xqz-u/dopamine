#! /usr/bin/bash

cd /data/$USER

# NOTE to install the setup:
# echo 'export SINGULARITY_CACHEDIR=/data/$USER/.singularity' >> ~/.bashrc
# singularity pull docker://mongo
# mkdir mongodb

singularity instance start --bind mongodb:/data/db mongo_latest.sif mongodb
