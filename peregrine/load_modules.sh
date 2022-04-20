#! /usr/bin/bash
# NOTE source this file, as it changes the $PATH

modules=('git/2.33.1-GCCcore-11.2.0-nodocs'
	 'Python/3.9.6-GCCcore-11.2.0')

for m in "${modules[@]}"
do
    module load "$m"
    echo "Loaded module $m"
done
