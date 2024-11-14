#!/bin/bash

#SBATCH --job-name=iemocap
#SBATCH --qos=a100-6
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/dilab/hrimlee/peft-ser/output.out

# python3 iemocap6_audio.py
python iemocap6.py
