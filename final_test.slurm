#!/bin/bash
#SBATCH --job-name=final-test-metric
#SBATCH --time=36:30:00
#SBATCH --output=results/final_test.txt
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#sBATCH --mem=50GB
#SBATCH --mail-type=END,FAIL
#notSBATCH --exclude=idyie,alpos,aura 


python final_test.py
