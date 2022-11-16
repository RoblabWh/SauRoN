#!/bin/sh

conda env create -f environment.yml

#default_env_name="sauron"

#read -p "Name for virtual environment (default: ${default_env_name}): " env_name

#if [ -z ${env_name} ]; then
#  env_name=${default_env_name}
#fi

#conda create --name ${env_name} --file requirements.txt --channel default --channel pytorch --channel conda-forge
