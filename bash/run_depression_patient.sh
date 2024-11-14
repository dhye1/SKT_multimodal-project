#!/bin/bash
#SBATCH --job-name=deprpa
#SBATCH --qos=a100-6
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/dilab/hrimlee/test/peft-ser/experiment_out/patient_wavlm.out

cd ../inference

dataset="daic"

num_epochs=30
hidden_dim=256
speaker=None 
modal='multimodal'
split_data_dir="/home/dilab/hrimlee/test/daic_woz_process/DAIC"

sym_dir=None
erc_dir=/home/dilab/hrimlee/test/peft-ser/finetune/iemocap6/multimodal/lr00005_ep30_lora_16
exp_name="Emotion_final_64.74" # pretained emotion weights dir 
phq_mode="depression"
txt_len=512

is_key_lora="True"         
cross_modal_atten="True"


batch_size=6
data_per_p=2


exp_dir="patient_test_${batch_size}_${data_per_p}"
python finetune_depression_symweight_simple_patient.py --text_model roberta-large --audio_model whisper-medium \
--dataset $dataset --split_data_dir $split_data_dir  \
--num_epochs $num_epochs --speaker $speaker --exp_dir $exp_dir --is_key_lora $is_key_lora \
--learning_rate 0.00003 --modal $modal --cross_modal_atten $cross_modal_atten \
--finetune_method 'lora' --finetune_roberta 'True' \
--lora_rank 16 --lora_alpha 16 --lora_dropout 0.1 --max_txt_len $txt_len \
--lora_target_modules "key","query","value" --phq_mode $phq_mode --erc_dir $erc_dir \
--exp_name $exp_name --sym_dir $sym_dir --data_per_patient $data_per_p --batch_size $batch_size
