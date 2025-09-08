#! /bin/bash
set -euxo pipefail

files=($(ls $1))

for key in "${files[@]}"
do
  uv run validate $1/$key
done
