#!/bin/bash

set -e

for value in $(seq 1 1 5)
do
echo $value


python evaluation.py -pred_file=Folder/Eval_out_new_$value.csv -ref_files_path=/home/gqw98/dcaseTask5/Development_Set/Validation_Set/ -team_name=EVAL_$value -dataset=VAL -savepath=./Folder/

done