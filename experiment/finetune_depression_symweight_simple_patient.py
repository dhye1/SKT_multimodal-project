import json
import yaml
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import loralib as lora
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'dataloader'))

from utils import parse_finetune_args, set_seed, log_epoch_result, log_best_result, excution_time, tokenize_texts, replace_report_labels

from utils import merge_lora_weights

from wav2vec import Wav2VecWrapper
from wavlm_plus2 import WavLMWrapper
from whisper_model6 import WhisperWrapper

from model.depression_prediction import  TextAudioClassifier
from evaluation import EvalMetric2

# patient dataset loader 
from dataloader_dep import load_finetune_audios, set_finetune_dataloader, return_weights

# patient roberta model  
from model.custom_roberta_p import RobertaCrossAttn

from transformers import RobertaTokenizer

from safetensors.torch import load_file 
from peft import PeftConfig, PeftModel, get_peft_model


'''
NOTE: [Last Update] 10/22
 
Depression finetune model 

symptom prediction 결과를 depression prediction에 활용할수 있도록 수정한 버전. 

'''

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Model hidden states information
hid_dim_dict = {
    "wav2vec2_0":       768,
    "tera":             768,
    "wavlm":            768,
    "roberta-base":     768,
    "roberta-large":    1024,
    "whisper-medium":   1024, 
    "whisper_small":    768,
    "whisper_base":     512,
    "whisper_tiny":     384,
    "apc":              512,
}

# Model number of encoding layers
num_enc_layers_dict = {
    "wav2vec2_0":       12,
    "wavlm":            12,
    "whisper_small":    12,
    "roberta-base":     12,
    "whisper_base":     6,
    "tera":             4,
    "whisper_tiny":     4,
    "apc":              3,
}

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 


def train_epoch(
    dataloader, 
    model, 
    device, 
    optimizer,
    weights,
    phq_mode
):
    model.train()
    symptom_criterion = nn.BCEWithLogitsLoss()
    depression_criterion = nn.BCEWithLogitsLoss()

    if phq_mode in ['symptom']:
        eval_metric = EvalMetric2(multilabel=True)
    else: 
        # depression
        eval_metric = EvalMetric2(multilabel=False)
    
    for batch_idx, batch_data in enumerate(dataloader):
        model.zero_grad()
        optimizer.zero_grad()
        
        x, x_text,speaker_id, phq_symptoms, phq_binary, length = batch_data 
        x, phq_symptoms, phq_binary = x.to(device), phq_symptoms.to(device).float(), phq_binary.to(device).float()
        
        # 
        phq_symptoms = phq_symptoms.mean(dim=1)  # Average over segments 
        phq_binary = phq_binary.mean(dim=1)  # Average over segments

        # print(f"Batch {batch_idx}:\n")
        # print(f"Audio Data Shape: {x.shape}")
        # print(f"Text Data: {x_text}")
        # print(f"Speaker IDs: {speaker_id}")
        # print(f"PHQ Symptoms: {phq_symptoms}")
        # print(f"PHQ Binary: {phq_binary}")
        # print(f"Audio Lengths: {length}\n")
        
        length = length.squeeze()
        
        if phq_mode in ['symptom']:
            symptom_outputs = model(audio_input=x, text_input=x_text, speaker_ID=speaker_id, length=length)
    
            # Compute loss per symptom
            symptom_losses = symptom_criterion(
                symptom_outputs, phq_symptoms, reduction='none'
            )  # Shape: [batch_size, num_symptoms]

            # Average loss over the batch for each symptom
            symptom_losses_per_symptom = symptom_losses.mean(dim=0)  # Shape: [num_symptoms]

            # Total symptom loss (average over all symptoms and batch)
            symptom_loss = symptom_losses_per_symptom.mean()
            
        
        elif phq_mode in ['depression']: 
            
            _ ,symptom_probs, depression_output = model(audio_input=x, text_input=x_text, speaker_ID=speaker_id, length=length)
            # Compute symptom loss
            symptom_loss = symptom_criterion(symptom_probs, phq_symptoms)
            # Compute depression loss
            depression_loss = depression_criterion(depression_output.squeeze(), phq_binary)
            # Total loss is the sum of both losses
            total_loss = symptom_loss + depression_loss
            
            print(f"total_loss: {total_loss} = symptom_loss: {symptom_loss} + depression_loss: {depression_loss}")
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        if phq_mode in ['symptom']:
            eval_metric.append_classification_results(phq_symptoms, symptom_outputs, total_loss)
            
        elif phq_mode in ['depression']: 
            eval_metric.append_classification_results(phq_binary, depression_output, total_loss)
        
        if (batch_idx % 10 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
            result_dict = eval_metric.classification_summary()
            logging.info(f'Fold {fold_idx} - Current Train Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["loss"]:.3f}')
            logging.info(f'Fold {fold_idx} - Current Train UAR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["uar"]:.2f}%')
            logging.info(f'Fold {fold_idx} - Current Train ACC at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["acc"]:.2f}%')
            logging.info(f'Fold {fold_idx} - Current Train LR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {scheduler.optimizer.param_groups[0]["lr"]}')
            logging.info(f'-------------------------------------------------------------------')

    logging.info(f'-------------------------------------------------------------------')
    result_dict = eval_metric.classification_summary()
    return result_dict


def validate_epoch(
    dataloader, 
    model, 
    device,
    weights,
    phq_mode,
    split:  str="Validation"
):  
    model.eval()
    
    symptom_criterion = nn.BCEWithLogitsLoss()
    depression_criterion = nn.BCEWithLogitsLoss()

    if phq_mode in ['symptom']:
        eval_metric = EvalMetric2(multilabel=True)
    else: # depression
        eval_metric = EvalMetric2(multilabel=False)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            x, x_text, speaker_id, phq_symptoms, phq_binary, length = batch_data 
            x, phq_symptoms, phq_binary = x.to(device), phq_symptoms.to(device).float(), phq_binary.to(device).float()
            # speaker_id = torch.tensor(speaker_id).to(device)
            
            phq_symptoms = phq_symptoms.mean(dim=1)  # Average over segments
            phq_binary = phq_binary.mean(dim=1)  # Average over segments
            
            phq_binary = phq_binary.unsqueeze(0) if phq_binary.dim() == 0 else phq_binary
            
            print(f"Batch {batch_idx}:\n")
            print(f"Audio Data Shape: {x.shape}")
            print(f"Text Data: {x_text}")
            print(f"Speaker IDs: {speaker_id}")
            print(f"PHQ Symptoms: {phq_symptoms}")
            print(f"PHQ Binary: {phq_binary}")
            print(f"Audio Lengths: {length}\n")
            if length.size(0) > 1:
                length = length.squeeze() 
            # length = torch.tensor([96000])
            # Forward pass: Get model outputs
            symptom_outputs = model(audio_input=x, text_input=x_text, speaker_ID=speaker_id, length=length)
            
            if phq_mode in ['symptom']:
                # Compute loss per symptom
                symptom_losses = symptom_criterion(
                    symptom_outputs, phq_symptoms, reduction='none'
                )  # Shape: [batch_size, num_symptoms]

                # Average loss over the batch for each symptom
                symptom_losses_per_symptom = symptom_losses.mean(dim=0)  # Shape: [num_symptoms]

                # Total symptom loss (average over all symptoms and batch)
                symptom_loss = symptom_losses_per_symptom.mean()
            
            elif phq_mode in ['depression']: 
                symptom_outputs, depression_output = symptom_outputs
                print(f"depression_output shape: {depression_output.shape}")
                print(f"phq_binary shape: {phq_binary.shape}")
            
                _, symptom_probs, depression_output = model(audio_input=x, text_input=x_text, speaker_ID=speaker_id, length=length)

              
                # Compute symptom loss
                symptom_loss = symptom_criterion(symptom_probs, phq_symptoms)
                # Compute depression loss
                if depression_output.shape == phq_binary.shape:
                    depression_loss = depression_criterion(depression_output, phq_binary)
                else:
                    # Ensure squeezing only removes a dimension if it exists
                    depression_loss = depression_criterion(depression_output.view_as(phq_binary), phq_binary)
                
                # Total loss is the sum of both losses
                total_loss = symptom_loss + depression_loss

                print(f"total_loss: {total_loss} = symptom_loss: {symptom_loss} + depression_loss: {depression_loss}")
            
            if phq_mode in ['symptom']:
                eval_metric.append_classification_results(phq_symptoms, symptom_outputs, total_loss)
                
            elif phq_mode in ['depression']: 
                eval_metric.append_classification_results(phq_binary, depression_output, total_loss)
            
            if (batch_idx % 50 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
                result_dict = eval_metric.classification_summary()
                logging.info(f'Fold {fold_idx} - Current {split} Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["loss"]:.3f}')
                logging.info(f'Fold {fold_idx} - Current {split} UAR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["uar"]:.2f}%')
                logging.info(f'Fold {fold_idx} - Current {split} ACC at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["acc"]:.2f}%')
                logging.info(f'-------------------------------------------------------------------')

    logging.info(f'-------------------------------------------------------------------')
    result_dict = eval_metric.classification_summary()
    logging.info(str(eval_metric.classification_summary()))
    if split == "Validation": scheduler.step(result_dict["loss"])
    return result_dict


if __name__ == '__main__':

    start_time = time.time()
    # Argument parser
    args = parse_finetune_args()
    print(args)
    
    with open("../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.split_dir  = str(Path(config["project_dir"]).joinpath(args.split_data_dir)) # for stt data inference 
    args.data_dir   = str(Path(config["project_dir"]).joinpath("audio"))
    args.log_dir    = str(Path(config["project_dir"]).joinpath("finetune"))

    print(" args.split_dir",  args.split_dir)
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    best_dict = dict()
    if args.dataset == "msp-improv": total_folds = 7
    elif args.dataset == "msp-podcast": total_folds = 4
    elif args.dataset in ["daic", "iemocap", "iemocap6", "meld", "meld6", "daic"]: total_folds = 2 # 
    else: total_folds = 6
    
    for fold_idx in range(1, total_folds):
    # Read train/dev file list
        train_file_list, dev_file_list, test_file_list = load_finetune_audios(
            args.split_dir, audio_path=args.data_dir, dataset=args.dataset, fold_idx=fold_idx
        )
        # Read weights of training data
        if args.dataset != 'daic':
            weights = return_weights(
                args.split_dir, dataset=args.dataset, fold_idx=fold_idx
            )
        
        # Set train/dev/test dataloader
        train_dataloader = set_finetune_dataloader(
            args, train_file_list, is_train=True
        )
        dev_dataloader = set_finetune_dataloader(
            args, dev_file_list, is_train=False
        )
        test_dataloader = set_finetune_dataloader(
            args, test_file_list, is_train=False
        )
        # Define log dir
        log_dir = Path(args.log_dir).joinpath(
            args.dataset, 
            args.modal, 
            args.setting
        )
        Path.mkdir(log_dir, parents=True, exist_ok=True)
        # Set seeds
        set_seed(8*fold_idx)
        
        if args.dataset in ["iemocap", "msp-improv", "iemocap_impro"]: num_class = 4
        elif args.dataset in ['iemocap6', 'meld6']: num_class = 6
        elif args.dataset in ["meld"]: num_class = 4
        
        elif args.dataset in ["msp-podcast"]: num_class = 4
        elif args.dataset in ["crema_d"]: num_class = 4
        elif args.dataset in ["ravdess"]: num_class = 7
        
        elif args.dataset in ['daic']: num_class = 2 
        

        ########### Representation Learning Model ###########
        if args.audio_model == "wav2vec2_0":
            audio_model = Wav2VecWrapper(args).to(device)
            
        elif args.audio_model == "wavlm_plus":
            audio_model = WavLMWrapper(args).to(device)
            audio_dim = hid_dim_dict[args.audio_model]
            
        elif args.audio_model in ["whisper_tiny", "whisper_base", "whisper_small", "whisper-medium", "whisper_large"]:
            audio_model = WhisperWrapper(args).to(device)
            if args.downstream: audio_dim = 256 
            else: audio_dim = hid_dim_dict[args.audio_model]

        if args.text_model in ["roberta-base", "roberta-large"]:
            if args.modal == 'multimodal': 
                text_model = RobertaCrossAttn(args, audio_model).to(device)
            else: 
                text_model = RobertaCrossAttn(args).to(device)

            text_dim = hid_dim_dict[args.text_model]
            tokenizer = RobertaTokenizer.from_pretrained(args.text_model)

        # Audio Modal
        if args.modal == 'audio':  
            text_model = None 
            text_dim   = None  
            
        # Text Modal        
        elif args.modal == 'text':
            audio_model = None 
            audio_dim   = None  
                
        ########### Speaker ID ###########
        if args.modal in ['audio', 'multimodal']:
            if args.speaker =='wavlm': 
                speaker_model = WavLMWrapper(args).to(device)
                speaker_dim = hid_dim_dict[args.speaker]
                # print("speaker_dim", speaker_dim)
                # print("speaker_model", speaker_model)
            elif args.speaker_dim is not None:
                speaker_model = None 
                speaker_dim = args.speaker_dim
            else: 
                speaker_model = None 
                speaker_dim = None # args.speaker_dim
        else:
            speaker_model = None 
            speaker_dim = None
            
        
        ########### Prediciton model ###########
        model = TextAudioClassifier(audio_model=audio_model ,text_model=text_model, speaker_model=speaker_model,\
                                    speaker=args.speaker, audio_dim=audio_dim, text_dim=text_dim, speaker_dim=speaker_dim, \
                                    hidden_dim=args.hidden_dim, num_classes=num_class, dropout_prob = args.dr, \
                                    cross_modal_atten = args.cross_modal_atten, modal = args.modal, phq_mode = args.phq_mode, num_symptom=5).to(device)

        
        # Load Pretrained ER model 
        if args.erc_dir != "None": 
            print("erc_dir", args.erc_dir)
            erc_dir = Path(args.erc_dir)
            
            ######### Load Pretrained Weights #########
            audio_model_path    = os.path.join(args.erc_dir, f'{args.exp_name}_whisper_merged_fold_1.pt')
            text_model_path     = os.path.join(args.erc_dir, f'{args.exp_name}_roberta_merged_fold_1.pt', 'model.safetensors') 
            speaker_model_path  = os.path.join(args.erc_dir, f'{args.exp_name}_whisper_fold_1.pt')
            
            # Load state dicts
            if model.speaker_model is not None:
                speaker_state_dict  = torch.load(speaker_model_path)
                model.speaker_model.load_state_dict(speaker_state_dict, strict=False)
            
            if model.text_model is not None:
                text_state_dict     = load_file(text_model_path)
                model.text_model.semantic_model.load_state_dict(text_state_dict, strict=False) # 
                
            if model.audio_model is not None:
                audio_state_dict    = torch.load(audio_model_path)
                print("whisper lora", audio_state_dict)
                model.audio_model.backbone_model.load_state_dict(audio_state_dict, strict=False)
            
        if args.sym_dir != "None":
            # exp_name is exp_dir of pretrained model 
            # sym_dir is save path of pretrained model 
            
            ######### Load Pretrained Weights #########
            audio_model_path    = os.path.join(args.sym_dir, f'{args.exp_name}_whisper_merged_fold_1.pt')
            text_model_path     = os.path.join(args.sym_dir, f'{args.exp_name}_roberta_merged_fold_1.pt', 'model.safetensors') 
            speaker_model_path  = os.path.join(args.sym_dir, f'{args.exp_name}_whisper_fold_1.pt')
            
            # Load state dicts
            if model.speaker_model is not None:
                speaker_state_dict  = torch.load(speaker_model_path)
                model.speaker_model.load_state_dict(speaker_state_dict, strict=False)
            
            if model.text_model is not None:
                text_state_dict     = load_file(text_model_path)
                model.text_model.semantic_model.load_state_dict(text_state_dict, strict=False)
                
            if model.audio_model is not None:
                audio_state_dict    = torch.load(audio_model_path)
                model.audio_model.backbone_model.load_state_dict(audio_state_dict, strict=False)
            
            # load PHQ-8 classifier 
            sym_model_path = os.path.join(args.sym_dir, f'{args.exp_name}_pred_fold_1.pt')
            sym_model_dict  = torch.load(sym_model_path)
            model.phq8_classifiers.load_state_dict(sym_model_dict, strict=False)
        
        if args.print_verbose:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    #print(f"{name}: {'trainable'}")
                    print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")      

        # Define the downstream models
        if args.downstream_model == "cnn":
            # Define the number of class
            if args.dataset in ["iemocap", "msp-improv",  "iemocap_impro"]: num_class = 4
            elif args.dataset in ["msp-podcast"]: num_class = 4
            elif args.dataset in ["crema_d"]: num_class = 4
            elif args.dataset in ["ravdess","meld"]: num_class = 7
            elif args.dataset in ["iemocap6","meld6"]: num_class = 6
        
        # Read trainable params
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f'Trainable params size: {params/(1e6):.2f} M')
        
        # Define optimizer
        optimizer = torch.optim.Adam(
            list(filter(lambda p: p.requires_grad, model.parameters())),
            lr=args.learning_rate, 
            weight_decay=1e-4,
            betas=(0.9, 0.98)
        )

        # Define scheduler, patient = 5, minimum learning rate 5e-5
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=5e-5
        )

        # Training steps
        best_dev_uar, best_test_uar, best_epoch = 0, 0, 0
        best_dev_acc, best_test_acc = 0, 0
        best_dev_mf1, best_test_mf1 = 0, 0
        
        result_hist_dict = dict()
        for epoch in range(args.num_epochs):
            train_result = train_epoch(
                train_dataloader, model, device, optimizer,weights=args.pos_weight, phq_mode =args.phq_mode
            )

            dev_result = validate_epoch(
                dev_dataloader, model, device,  weights=args.pos_weight,phq_mode=args.phq_mode
            )
            
            test_result = validate_epoch(
                test_dataloader, model, device, weights=args.pos_weight, split="Test", phq_mode=args.phq_mode
            )
            logging.info(f'Before update - Best epoch: {best_epoch}, Best test acc: {best_test_acc}')

            # if we get a better results
            if best_dev_mf1 < dev_result["mf1"]: 
                
                best_dev_uar = dev_result["uar"]
                best_test_uar = test_result["uar"]

                best_dev_acc = dev_result["acc"]
                best_test_acc = test_result["acc"]

                best_dev_mf1 = dev_result["mf1"]
                best_test_mf1 = test_result["mf1"]

                best_dev_report = dev_result["report"]
                best_test_report = test_result["report"]
                
                best_epoch = epoch
                            
                if args.modal in ['multimodal', 'multimodal_concat']: 
                    torch.save(model.phq8_classifiers.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_pred_fold_{fold_idx}.pt')))
                        
                    # Whisper save LoRA  
                    if args.finetune_method == "lora" or args.finetune_method == "combined"  or args.finetune_method == "lora_all" \
                    or args.finetune_method == "all" or args.finetune_method == "lora_attn":
                        torch.save(lora.lora_state_dict(model), str(log_dir.joinpath(f'{args.exp_dir}_whisper_fold_{fold_idx}.pt')))
                        print(f"Model save: {args.finetune_method}")
                    
                    # Whisper save model 
                    elif args.finetune_method == "adapter" or args.finetune_method == 'True' or args.finetune_method == 'False':
                        torch.save(model.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_{args.finetune_method}_whisper_fold_{fold_idx}.pt')))
                        print(f"Model save: {args.finetune_method}")
                        
                    # Roberta PEFT save  
                    if args.finetune_roberta:
                        model.text_model.semantic_model.save_pretrained(str(log_dir.joinpath(f'{args.exp_dir}_roberta_fold_{fold_idx}.pt'))) 
                        
                elif args.modal in ['audio']: 
                    torch.save(model.phq8_classifiers.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_pred_fold_{fold_idx}.pt')))
                        
                    # Whisper save LoRA  
                    if args.finetune_method == "lora" or args.finetune_method == "combined"  or args.finetune_method == "lora_all" \
                    or args.finetune_method == "all" or args.finetune_method == "lora_attn":
                        torch.save(lora.lora_state_dict(model), str(log_dir.joinpath(f'{args.exp_dir}_whisper_fold_{fold_idx}.pt')))

                    # Whisper save model 
                    elif args.finetune_method == "adapter" or args.finetune_method == 'True' or args.finetune_method == 'False':
                        torch.save(model.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_{args.finetune_method}_whisper_fold_{fold_idx}.pt')))
                    print(f"Model save: {args.finetune_method}")
                        
                elif args.modal in ['text']:  
                    torch.save(model.phq8_classifiers.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_pred_fold_{fold_idx}.pt')))      
                    # Roberta PEFT save  
                    if args.finetune_roberta:
                        model.text_model.semantic_model.save_pretrained(str(log_dir.joinpath(f'{args.exp_dir}_roberta_fold_{fold_idx}.pt')), save_adapters=True) 
                        print(f"Model save: finetune_roberta [{args.finetune_roberta}] ")
                
                if args.phq_mode in ['depression']:
                    torch.save(model.phq8_binary_classifier.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_classifier_fold_{fold_idx}.pt')))
                    torch.save(model.W_d.state_dict(), str(log_dir.joinpath(f'{args.exp_dir}_Wd_fold_{fold_idx}.pt')))
                    torch.save(model.b_d, str(log_dir.joinpath(f'{args.exp_dir}_bd_fold_{fold_idx}.pt')))          
                    
                
            # Load best model and merge lora weights. (base model + LORA)
            if epoch == args.num_epochs-1:
        
                # Whisper save LoRA  
                if args.modal in ['audio', 'multimodal', 'multimodal_concat']:
                    if args.finetune_method == "lora" or args.finetune_method == "combined"  or args.finetune_method == "lora_all" \
                    or args.finetune_method == "all" or args.finetune_method == "lora_attn":
                        state_dict =  model.audio_model.org_state_dict
                        
                        whisper_path = str(log_dir.joinpath(f'{args.exp_dir}_whisper_fold_{fold_idx}.pt'))
                        audio_state_dict = torch.load(whisper_path)
                        
                        merged_state_dict = merge_lora_weights(state_dict, audio_state_dict, args.lora_rank)
                        torch.save(merged_state_dict, str(log_dir.joinpath(f'{args.exp_dir}_whisper_merged_fold_{fold_idx}.pt')), _use_new_zipfile_serialization=False)

                if args.modal in ['text', 'multimodal', 'multimodal_concat']:
                    # Roberta PEFT save 
                    if args.finetune_roberta and model.text_model is not None :
                        merged_model = model.text_model.semantic_model.merge_and_unload()
                        merged_model.save_pretrained(str(log_dir.joinpath(f'{args.exp_dir}_roberta_merged_fold_{fold_idx}.pt')), save_adapters=True, save_embedding_layers=True, safe_serialization=True)
                        print(f"Merged Model save: finetune_roberta [{args.finetune_roberta}] ")
            
            logging.info(f'-------------------------------------------------------------------')
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev UAR {best_dev_uar:.2f}%, best test UAR {best_test_uar:.2f}%")
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev F1 {best_dev_mf1:.2f}%, best test F1 {best_test_mf1:.2f}%")
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev ACC {best_dev_acc:.2f}%, best test ACC {best_test_acc:.2f}%")
            logging.info(f'-------------------------------------------------------------------')
            
            # log the current result
            log_epoch_result(result_hist_dict, epoch, train_result, dev_result, test_result, log_dir, fold_idx, args.exp_dir)

        # log the best results
        log_best_result(result_hist_dict, epoch, best_dev_uar, best_dev_acc, best_test_uar, best_test_acc, log_dir, fold_idx, args.exp_dir)
        
        best_dict[fold_idx] = dict()
        best_dict[fold_idx]["mf1"]    = best_test_mf1
        best_dict[fold_idx]["uar"]    = best_test_uar
        best_dict[fold_idx]["acc"]    = best_test_acc
        best_dict[fold_idx]["report"] = best_test_report
        
        if args.phq_mode in ['symptom']:
            
            # sym = ['nointerest', 'depressed', 'sleep', 'tired', 'appetite', 'failure', 'concentrating', 'moving']
            sym = ['nointerest', 'depressed', 'sleep', 'tired', 'failure']

            for i in range(len(best_result_list_wf1)):
                acc_key = f'acc_{sym[i]}'
                mf1_key = f'mf1_{sym[i]}'
                best_dict[fold_idx][acc_key] = best_result_list_acc[i]
                best_dict[fold_idx][mf1_key] = best_result_list_wf1[i]
                                            
        # save best results
        jsonString = json.dumps(best_dict, indent=4)
        jsonFile = open(str(log_dir.joinpath(f'{args.exp_dir}_results.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    uar_list = [best_dict[fold_idx]["uar"] for fold_idx in best_dict]
    mf1_list = [best_dict[fold_idx]["mf1"] for fold_idx in best_dict]
    acc_list = [best_dict[fold_idx]["acc"] for fold_idx in best_dict]
    
    best_dict[fold_idx]["report"] = replace_report_labels(best_dict[fold_idx]["report"], args)

    best_dict["average"] = dict()
    best_dict["average"]["mf1"] = np.mean(mf1_list)
    best_dict["average"]["uar"] = np.mean(uar_list)
    best_dict["average"]["acc"] = np.mean(acc_list)
    
    best_dict["std"] = dict()
    best_dict["std"]["mf1"] = np.std(mf1_list)
    best_dict["std"]["uar"] = np.std(uar_list)
    best_dict["std"]["acc"] = np.std(acc_list)
    
    end_time = time.time()
    
    # save best results
    jsonString = json.dumps(best_dict, indent=4)
    jsonFile = open(str(log_dir.joinpath(f'{args.exp_dir}_results.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.write(f'Trainable params size: {params/(1e6):.2f} M')
    jsonFile.write(excution_time(start_time, end_time))
    jsonFile.close()