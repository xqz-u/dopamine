#! /usr/bin/bash

# TODO mongod config:
# - increase snapshot time (~2m rn?)

# NOTE the singularity image should be present on the host; it can
# be built locally with:
# sudo singularity build mongo_latest.sif ../resources/data/mongodb/Singularity

cd /data/$USER
# to install the setup:
# echo 'export SINGULARITY_CACHEDIR=/data/$USER/.singularity' >> ~/.bashrc
# singularity pull docker://mongo
# mkdir mongodb mongodb_logs

singularity instance start \
	    -B mongodb:/data/db \
	    -B $HOME/thesis/resources/data/mongodb/mongoconf:/etc/mongo/ \
	    -B mongodb_logs:/var/log/mongodb \
	    mongo_latest.sif mongodb
