#!/bin/bash

args=""
# args="${args} --clahe"
args="${args} --intensity"
args="${args} --flip"
# args="${args} --rotate"
args="${args} --jawratio"
formatted_args=${args//--/_}
formatted_args=${formatted_args// /}
formatted_args=${formatted_args#_}
if [ -z "$formatted_args" ]
then
    formatted_args="no_aug"
fi
formatted_args="${formatted_args}"


num_classes_values=(3)
label_ratio_values=(0 5 10 50 100)
for num_classes in "${num_classes_values[@]}"; do
  for label_ratio in "${label_ratio_values[@]}"; do
    cmd="python trainer.py ${args} --num_classes ${num_classes} --label_ratio ${label_ratio}"
    # echo $cmd
    eval $cmd
  done
done