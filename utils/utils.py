import json
import torch
import random
import numpy as np
import transformers
import argparse, logging

# depression args 추가 

transformers.logging.set_verbosity(40)

logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


def log_gpu_memory_usage(log_dict, phase="inference"):
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # bytes to GB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)  # bytes to GB

    # 로그에 메모리 사용량 기록
    log_dict[f"{phase}_allocated_memory_GB"] = allocated_memory
    log_dict[f"{phase}_reserved_memory_GB"] = reserved_memory

    # 콘솔에 메모리 사용량 출력
    print(f"{phase} Allocated memory: {allocated_memory:.4f} GB")
    print(f"{phase} Reserved memory: {reserved_memory:.4f} GB")



def add_prefix_to_keys(state_dict, prefix):
    new_state_dict = {}
    for key in state_dict.keys():
        # Check if the key starts with the embeddings, encoder, or other relevant layers
        if any(key.startswith(layer) for layer in ["embeddings", "encoder", "pooler"]):
            new_key = prefix + key
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def merge_lora_weights(base_state_dict, lora_state_dict, rank=16):
    '''
    merge lora weight for Whisper

    - base_state_dict: pretrained weight
    - lora state dict: lora weight  
    '''
    
    # encoder.layers.4.fc2.lora_A
    
    
    # Iterate through all LoRA keys to find and merge those starting with 'audio_model.backbone_model'
    for key in lora_state_dict.keys():
        if key.startswith('audio_model.backbone_model') and 'lora_A' in key:
            # Get the corresponding lora_B key
            lora_A_key = key
            lora_B_key = key.replace('lora_A', 'lora_B')

            if lora_B_key in lora_state_dict:
                # Extract lora_A and lora_B
                lora_A = lora_state_dict[lora_A_key].cuda()
                lora_B = lora_state_dict[lora_B_key].cuda()
                # print(f"Shape mismatch: lora_B.shape = [{lora_B_key}] {lora_B.shape}, [{lora_A_key}] lora_A.shape = {lora_A.shape}")

                delta_W = (lora_B @ lora_A) * (1 / rank) # (1 / lora_rank)
            
                # Remove the 'audio_model.backbone_model.' prefix
                base_weight_key = lora_A_key.replace('audio_model.backbone_model.', '').replace('.lora_A', '.weight')

                if base_weight_key in base_state_dict:
                    print(f"Merging LoRA into {base_weight_key}")
                    base_state_dict[base_weight_key] = base_state_dict[base_weight_key].cuda()
                    # Add delta_W to the base weights
                    base_state_dict[base_weight_key] += delta_W
                else:
                    print(f"Base weight key {base_weight_key} not found in base model")
            else:
                print(f"Matching lora_B not found for {lora_A_key}")

    return base_state_dict


label_dict = {
    "iemocap":   {"0": "neu", "1": "sad", "2": "ang", "3": "hap"},
    "meld":      {"0": "neutral", "1": "sadness", "2": "anger", "3": "joy"},
    "iemocap6":  {"0": "neu", "1": "sad", "2": "fru", "3": "ang", "4": "hap", "5": "exc"},
    "meld6":     {"0": "neutral", "1": "sadness", "2": "anger", "3": "joy", "4": "surprise", "5": "fear", "6": "disgust"},
    
    "symptom":   {"0": "phq_nointerest", "1": "phq_depressed", "2": "phq_sleep", "3": "phq_tired", "4": "phq_appetite", \
                  "5": "phq_failure", "6": "phq_concentrating", "7": "phq_moving"},
    "depression": {"0": "non", "1": "dep"},
}
        
def replace_report_labels(report, args):
    '''
    Saving Classificatoin report with class-wise result 
    '''
    if args.dataset != 'daic':
        label_map = label_dict[args.dataset]
    else:
        label_map = label_dict[args.phq_mode]
    print(label_map)
    new_report = {}
    for key, value in report.items():
        if key in label_map:
            new_key = label_map[key]
            new_report[new_key] = value
        else:
            new_report[key] = value
    return new_report

def flat_text(text):
    # 리스트 내부에 두 문장이 포함되어있을경우 (e.g.,  [ '문장1' '문장2' ] ) 처리 
    flat_texts = []
    for sublist in text:
        if isinstance(sublist, list):
            if len(sublist) == 1:
                flat_texts.extend(sublist)
            else: # 두 문장인 경우 
                flat_texts.append(" ".join(sublist))
        else:
            flat_texts.append(sublist)
    return flat_texts


def tokenize_texts(texts, tokenizer, max_txt_len=32, device='cuda'):
    flat_txt = flat_text(texts)
    encodings = tokenizer(flat_txt, padding=True, truncation=True, max_length=max_txt_len, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    return input_ids, attention_mask


def flat_text_p(text):
    """
    Flatten deeply nested lists of text while ensuring that each conversation (utterance) is processed individually.
    The input can be deeply nested lists of text, such as:
    [[text1, text2, ...], [text3, text4, ...], ...]
    """
    flat_texts = []
    for sublist in text:
        # Iterate over each outer list
        if isinstance(sublist, list):
            # Handle cases where the sublist contains lists of texts
            joined_texts = []
            for inner_text in sublist:
                if isinstance(inner_text, str):
                    # Join each inner string if it's already a string
                    joined_texts.append(inner_text)
                elif isinstance(inner_text, list):
                    # If there's deeper nesting, join that
                    joined_texts.append(" ".join(inner_text))
            # After handling inner structure, join the entire conversation
            flat_texts.append(" ".join(joined_texts))
        else:
            flat_texts.append(sublist)
    return flat_texts

def tokenize_texts_p(texts, tokenizer, max_txt_len=32, device='cuda'):
    """
    Tokenize texts, including handling of nested lists. 
    The input is expected to be nested lists of strings. Each passage is tokenized separately.
    """
    # Flatten the nested text while preserving individual conversations or utterances
    flat_txt = flat_text_p(texts)

    # Tokenize using the provided tokenizer
    encodings = tokenizer(
        flat_txt, 
        padding=True, 
        truncation=True, 
        max_length=max_txt_len, 
        return_tensors="pt"
    )

    # Move tensors to the correct device
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)

    return input_ids, attention_mask

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def excution_time(start_time, end_time):
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    duration_string = f"Duration: {hours} hours, {minutes} minutes, {seconds} seconds"
    return duration_string

def set_seed(seed):
    print(f"seed: {seed}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def get_results(input_dict):
    return_dict = dict()
    return_dict["mf1"] = input_dict["mf1"]
    return_dict["uar"] = input_dict["uar"]
    return_dict["acc"] = input_dict["acc"]
    return_dict["loss"] = input_dict["loss"]
    return return_dict

def log_epoch_result(
    result_hist_dict:       dict, 
    epoch:                  int,
    train_result:           dict,
    dev_result:             dict,
    test_result:            dict,
    log_dir:                str,
    fold_idx:               int,
    exp_dir:                str,
):
    # read result
    result_hist_dict[epoch] = dict()
    result_hist_dict[epoch]["train"] = get_results(train_result)
    result_hist_dict[epoch]["dev"] = get_results(dev_result)
    result_hist_dict[epoch]["test"] = get_results(test_result)
    
    # dump the dictionary
    jsonString = json.dumps(result_hist_dict, indent=4)
    #jsonFile = open(str(log_dir.joinpath(f'{exp_dir}_fold_{fold_idx}.json')), "w")
    jsonFile = open(str(log_dir.joinpath(f'fold_{fold_idx}.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()
    
    
def log_best_result(
    result_hist_dict:       dict, 
    epoch:                  int,
    best_dev_uar:           float,
    best_dev_acc:           float,
    best_test_uar:          float,
    best_test_acc:          float,
    log_dir:                str,
    fold_idx:               int,
    exp_dir:                str
):
    # log best result
    result_hist_dict["best"] = dict()
    result_hist_dict["best"]["dev"], result_hist_dict["best"]["test"] = dict(), dict()
    result_hist_dict["best"]["dev"]["uar"] = best_dev_uar
    result_hist_dict["best"]["dev"]["acc"] = best_dev_acc
    result_hist_dict["best"]["test"]["uar"] = best_test_uar
    result_hist_dict["best"]["test"]["acc"] = best_test_acc

    # save results for this fold
    jsonString = json.dumps(result_hist_dict, indent=4)
    #jsonFile = open(str(log_dir.joinpath(f'{exp_dir}_fold_{fold_idx}.json')), "w")
    jsonFile = open(str(log_dir.joinpath(f'fold_{fold_idx}.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

def parse_finetune_args():
    # parser
    parser = argparse.ArgumentParser(description='emo2vec finetune experiments')
    parser.add_argument(
        '--data_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/audio',
        type=str, 
        help='raw audio path'
    )

    parser.add_argument(
        '--model_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/model',
        type=str, 
        help='model save path'
    )

    parser.add_argument(
        '--split_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/train_split',
        type=str, 
        help='train split path'
    )
    parser.add_argument(
        '--split_data_dir', 
        default='train_split',
        type=str, 
        help='train split path'
    )
    parser.add_argument(
        '--log_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/finetune',
        type=str, 
        help='model save path'
    )

    parser.add_argument(
        '--uar_dir', 
        default='/media/data/projects/speech-privacy/trust-ser/uar',
        type=str, 
        help='model uar history'
    )

    parser.add_argument(
        '--pretrain_model', 
        default='wav2vec2_0',
        type=str,
        help="pretrained model type"
    )

    parser.add_argument(
        '--finetune', 
        default='frozen',
        type=str,
        help="partial finetune or not"
    )
    
    parser.add_argument(
        '--learning_rate', 
        default=0.0002,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        '--num_epochs', 
        default=50,
        type=int,
        help="total training rounds",
    )
    
    parser.add_argument(
        '--optimizer', 
        default='adam',
        type=str,
        help="optimizer",
    )
    
    parser.add_argument(
        '--dataset',
        default="iemocap",
        type=str,
        help="Dataset name",
    )
    
    parser.add_argument(
        '--audio_duration', 
        default=6,
        type=int,
        help="audio length for training"
    )

    parser.add_argument(
        '--downstream_model', 
        default='rnn',
        type=str,
        help="model type"
    )

    parser.add_argument(
        '--num_layers',
        default=1,
        type=int,
        help="num of layers",
    )

    parser.add_argument(
        '--snr',
        default=45,
        type=int,
        help="SNR of the audio",
    )

    parser.add_argument(
        '--conv_layers',
        default=3,
        type=int,
        help="num of conv layers",
    )

    parser.add_argument(
        '--hidden_size',
        default=256,
        type=int,
        help="hidden size",
    )

    parser.add_argument(
        '--pooling',
        default='att',
        type=str,
        help="pooling method: att, average",
    )

    parser.add_argument(
        '--norm',
        default='nonorm',
        type=str,
        help="normalization or not",
    )
    
    parser.add_argument(
        '--finetune_method', 
        default='finetune',
        type=str, 
        help='finetune method: lora, lora_attn, lora_all, all, adapter'
    )
    
    parser.add_argument(
        '--adapter_hidden_dim', 
        default=128,
        type=int, 
        help='adapter dimension'
    )
    
    parser.add_argument(
        '--finetune_emb', 
        default="all",
        type=str, 
        help='adapter dimension'
    )
    
    parser.add_argument(
        '--embedding_prompt_dim', 
        default=5,
        type=int, 
        help='adapter dimension'
    )
    
    parser.add_argument(
        '--lora_rank', 
        default=16,
        type=int, 
        help='lora rank'
    )

    parser.add_argument(
        '--hidden_dim', 
        default=256,
        type=int, 
        help='hidden dim of prediction model  '
    )

    parser.add_argument(
        '--use-conv-output', 
        action='store_true',
        help='use conv output'
    )
    
    parser.add_argument(
        '--speaker', 
        default="None",
        type=str, 
        help='speaker emb type: [speaker_emb, wavlm, speaker_emb_0, None]'
    )
    
    parser.add_argument(
        '--downstream', 
        default=False,
        type=bool, 
        help='Flag to use downstream model'
    )
    
    parser.add_argument(
        '--exp_dir', 
        default="exp",
        type=str, 
        help='Exp dir'
    )
    
    parser.add_argument(
        '--max_audio_len',  # DON'T CHANGE 
        default=6,
        type=int, 
        help='max_audio_len'
    )

    parser.add_argument(
        '--max_txt_len', 
        default=128, 
        type=int, 
        help='max_txt_len'
    )
    
    parser.add_argument(
        '--pretrain_path', 
        default="lora_16_wavlm_frozen_fold_5",
        type=str, 
        help='pretrain model path'
    )
    
    parser.add_argument(
        '--inference_path', 
        default="stt_inference",
        type=str, 
        help='inference_path_name'
    )

    parser.add_argument(
        '--is_key_lora', 
        default=True,
        type=str2bool, 
        help='lora at key'
    )
   
    parser.add_argument(
        '--ws', 
        default=False,
        type=str2bool, 
        help='weighted sum feature'
    )

    parser.add_argument(
        '--wg', 
        default=False,
        type=str2bool, 
        help='weighted gate'
    )
    
    parser.add_argument(
        '--cross_modal_atten', 
        default=False,
        type=str2bool, 
        help='cross-modal (audio, text) attention'
    )

    parser.add_argument(
        '--modal', 
        default="audio",
        type=str, 
        help='[audio, text, multimodal]'
    )
    
    parser.add_argument(
        '--audio_model', 
        default="None",
        type=str, 
        help="Audio Modal Representation model"
    )
    
    parser.add_argument(
        '--text_model', 
        default="None",
        type=str, 
        help='Text Modal Representation model'
    )

    parser.add_argument(
        '--print_verbose', 
        default=True,
        type=str2bool, 
        help='print verbose'
    )

    # LoRA 관련 인자들 추가
    parser.add_argument(
        '--lora_alpha', 
        default=16, 
        type=int, 
        help='LoRA alpha for weight scaling'
    )

    parser.add_argument(
        '--lora_dropout', 
        default=0.1, 
        type=float, 
        help='Dropout probability for LoRA layers'
    )

    parser.add_argument(
        '--lora_target_modules', 
        default="dense", 
        type=str, 
        help='List of target modules for LoRA'
    )

    parser.add_argument(
        '--finetune_roberta', 
        default=True,
        type=str2bool, 
        help='finetune roberta'
    )
    
    parser.add_argument(
        '--dr', 
        default=0.5,
        type=float, 
        help='dropout ratio'
    )
    
    parser.add_argument(
        '--self_attn', 
        default=False,
        type=str2bool, 
        help='self atten before cross modal attn'
    )

    parser.add_argument(
        '--num_hidden_layers', 
        default=None,
        type=int, 
        help='number of hidden layers using repersentation, if none, all layers are used'
    )
    
    parser.add_argument(
        '--inference_mode', 
        default=False,
        type=str2bool, 
        help='inference or train'
    )
    
    
    ##### depression 
    parser.add_argument(
        '--use_pretrained_emotion', 
        default=False,
        type=str2bool, 
        help='whether to use pretrained emotion model'
    )

    parser.add_argument(
        '--phq_mode', 
        default='symptom',
        type=str, 
        help='[symptom or depression]'
    )
    
    parser.add_argument(
        '--erc_dir', 
        default=None,
        type=str, 
        help='pretrained ERC model dir'
    )
    
    parser.add_argument(
        '--sym_dir', 
        default="None",
        type=str, 
        help='pretrained symptom identification model dir'
    )

    parser.add_argument(
        '--exp_name', 
        default="None",
        type=str, 
        help='pretrained model exp_dir'
    )
    
    # patient wise learning
    parser.add_argument(
        '--data_per_patient', 
        default=4,
        type=int, 
        help='number of interview segments per patient'
    )
    
    parser.add_argument(
        '--batch_size', 
        default=4,
        type=int, 
        help='number of interview segments per patient'
    )

    parser.add_argument(
        '--random_seed', 
        default=1,
        type=int, 
        help='seed number'
    )

   
    args = parser.parse_args()
    if args.finetune_method == "adapter" or args.finetune_method == "adapter_l":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}_{args.adapter_hidden_dim}'
    elif args.finetune_method == "embedding_prompt":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}_{args.embedding_prompt_dim}'
    elif args.finetune_method == "lora" or "lora_attn" or "lora_all":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}_{args.lora_rank}'
    elif args.finetune_method == "finetune":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}'
    elif args.finetune_method == "combined":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}_{args.adapter_hidden_dim}_{args.embedding_prompt_dim}_{args.lora_rank}'
    elif args.finetune_method == "all":
        setting = f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.finetune_method}_{args.adapter_hidden_dim}_{args.lora_rank}'
    args.setting = setting
    if args.finetune_emb != "all":
        args.setting = args.setting + "_avgtok"
    if args.use_conv_output:
        args.setting = args.setting + "_conv_output"
    
    if args.dataset == 'iemocap':
        args.speaker_dim = 10
    elif args.dataset =='meld':
        args.speaker_dim = 260
    elif args.dataset == 'daic':
        args.speaker_dim = None
    
    return args

