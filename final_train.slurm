#!/bin/bash
#SBATCH --job-name=final-train-metric
#SBATCH --time=7-16:30:00
#SBATCH --output=results/final_train.txt
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#sBATCH --mem=512GB
#SBATCH --mail-type=END,FAIL
#notSBATCH --exclude=idyie,alpos,aura 



python final_train.py
