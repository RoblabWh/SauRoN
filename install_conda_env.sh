#!/bin/sh

default_env_name="sauron"

read -p "Name for virtual environment (default: ${default_env_name}): " env_name

if [ -z ${env_name} ]; then
  env_name=${default_env_name}
fi

conda env update --name "${env_name}" --file "environment.yml"
