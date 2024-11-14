#!/bin/bash
#SBATCH --job-name=daic
#SBATCH --qos=a100-6
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/dilab/hrimlee/test/peft-ser/experiment_out/daic_inference.out

cd ../inference

modal='multimodal'
dataset=daic

num_epochs=30
hidden_dim=256
speaker=None

split_data_dir="/home/dilab/hrimlee/test/daic_woz_process/DAIC"
txt_len=512
phq_mode="depression"

# pretrained weight name  
erc_dir=None
sym_dir=/home/dilab/hrimlee/test/peft-ser/finetune/daic/multimodal/lr3e-05_ep30_lora_16

exp_name="patient_batch_6_2"

batch_size=6
data_per_p=2

cross_modal_atten="True"
is_key_lora="True"

# save name 
exp_dir="dep_1024_crossattn(True)_ffn_(kqv)" 

python emotion_inference_daic.py --text_model roberta-large --audio_model whisper-medium \
    --dataset $dataset --inference_mode 'True' \
    --num_epochs $num_epochs --speaker $speaker --exp_dir $exp_dir --is_key_lora $is_key_lora \
    --learning_rate 0.00003 --modal $modal --cross_modal_atten $cross_modal_atten \
    --finetune_method 'lora' --finetune_roberta 'True' --split_data_dir $split_data_dir --max_txt_len $txt_len \
    --lora_rank 16 --lora_alpha 16 --lora_dropout 0.1 --lora_target_modules "key","query","value" \
    --phq_mode $phq_mode --erc_dir $erc_dir --exp_name $exp_name --sym_dir $sym_dir --data_per_patient $data_per_p 
