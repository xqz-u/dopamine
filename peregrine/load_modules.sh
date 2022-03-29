#! /usr/bin/bash
# NOTE source this file, as it changes the $PATH

modules=('Python/3.9.6-GCCcore-11.2.0'
	 'git/2.32.0-GCCcore-10.3.0-nodocs')

for m in "${modules[@]}"
do
    module load "$m"
    echo "Loaded module $m"
done
