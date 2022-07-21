#!/bin/bash

set -e

for value in $(seq 1 1 5)
do
echo $value


python post_proc.py -val_path=/home/gqw98/dcaseTask5/Development_Set/Validation_Set/ -evaluation_file=Eval_out_$value.csv -new_evaluation_file=Eval_out_new_$value.csv

done