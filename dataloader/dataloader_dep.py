import json
import copy
import glob
import torch
import random
import torchaudio
import numpy as np
import pandas as pd
import pickle, pdb, re

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from audiomentations import Compose, AddBackgroundNoise, PolarityInversion, AddGaussianSNR, TimeMask, TimeStretch

import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict


'''
NOTE: 
- last update: 2024.10.10

- PatientDepressionDatasetGenerator 
    (환자 id별로 데이터 그룹핑 후, data_per_patient 만큼 데이터 랜덤 추출)
- collate function 3 추가 (collate function for PatientDepressionDatasetGenerator)

'''

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


import torch

def collate_fn3(batch):
    # Unpack the batch into its components
    audio_batch, text_batch, speaker_id_batch, phq_symptoms_batch, phq_binary_batch, len_batch = zip(*batch)
    
    # Find the minimum length across all audio tensors
    min_audio_len = min([a.shape[0] for a in audio_batch])  # Trim to the shortest audio length

    # Trim the audio_batch tensors to the minimum length
    audio_batch = [a[:min_audio_len] for a in audio_batch]
    
    # Stack the trimmed audio_batch tensors into a single tensor
    audio_batch = torch.stack(audio_batch)

    # Find the minimum length across phq_binary_batch and phq_symptoms_batch
    min_len_binary = min([len(b) for b in phq_binary_batch])
    min_len_symptoms = min([s.size(0) for s in phq_symptoms_batch])
    min_len = min(min_len_binary, min_len_symptoms)

    # Trim the phq_binary_batch tensors to the minimum length
    phq_binary_batch = [b[:min_len] for b in phq_binary_batch]
    phq_binary_batch = torch.stack(phq_binary_batch)

    # Trim the phq_symptoms_batch tensors to the minimum length
    phq_symptoms_batch = [s[:min_len] for s in phq_symptoms_batch]
    phq_symptoms_batch = torch.stack(phq_symptoms_batch)

    # Trim len_batch to the minimum length
    len_batch = [l[:min_len] for l in len_batch]
    len_batch = torch.stack(len_batch)

    return audio_batch, text_batch, speaker_id_batch, phq_symptoms_batch, phq_binary_batch, len_batch




def padding_cropping(input_wav, size):
    if len(input_wav) > size:
        input_wav = input_wav[:size]
    elif len(input_wav) < size:
        input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
    return input_wav



def collate_fn2(batch):
    
    '''
    collate function for daic dataset 
    '''
    
    max_audio_len = min(max([b[0].shape[0] for b in batch]), 16000 * 6)
    
    data, text_data, speaker_id, phq_symptoms, phq_binary, len_data = [], [], [], [], [], []

    for idx in range(len(batch)):
        # Append audio data
        data.append(padding_cropping(batch[idx][0], max_audio_len))
        text_data.append(batch[idx][1])
        speaker_id.append(batch[idx][2])

        # Append target labels (symptom labels and binary labels)
        phq_symptoms.append(batch[idx][3])
        phq_binary.append(torch.tensor(batch[idx][4]))

        # Append length
        if len(batch[idx][0]) >= max_audio_len:
            len_data.append(torch.tensor(max_audio_len))
        else:
            len_data.append(torch.tensor(len(batch[idx][0])))

    data = torch.stack(data, dim=0)
    len_data = torch.stack(len_data, dim=0)
    phq_symptoms = torch.stack(phq_symptoms, dim=0)
    phq_binary = torch.stack(phq_binary, dim=0)

    return data, text_data, speaker_id, phq_symptoms, phq_binary, len_data

class PatientDepressionDatasetGenerator(Dataset):
    def __init__(
        self,
        data_list: list,
        noise_list: list,
        data_len: int,
        is_train: bool = False,
        audio_duration: int = 6,
        model_type: str = "rnn",
        apply_guassian_noise: bool = False,
        dataset: str = 'iemocap',
        data_per_patient: int = 4
    ):
        """
        Set dataloader for depression finetuning.
        :param data_list:       Audio list files
        :param noise_list:      Audio list files
        :param data_len:        Length of input audio file size
        :param is_train:        Flag for dataloader, True for training; False for dev
        :param audio_duration:  Max length for the audio length
        :param model_type:      Type of the model
        :param data_per_patient: Number of data samples per patient
        """
        self.data_list = data_list
        self.noise_list = noise_list
        self.data_len = data_len
        self.is_train = is_train
        self.audio_duration = audio_duration
        self.model_type = model_type
        self.apply_guassian_noise = apply_guassian_noise
        self.data = dataset
        self.data_per_patient = data_per_patient

        self.transform = Compose([
            AddGaussianSNR(min_snr_in_db=10.0, max_snr_in_db=30.0, p=1.0),
            TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
        ])

        # Group data by patient ID
        self.patient_data = defaultdict(list)
        for item in self.data_list:
            patient_id = item[0]
            self.patient_data[patient_id].append(item)

        # Create unique list of patient IDs
        self.patient_ids = list(set(item[0] for item in self.data_list))
        
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        # Get patient ID based on index
        patient_id = self.patient_ids[index]
        patient_records = self.patient_data[patient_id]
        
        # Sample `self.data_per_patient` records for the patient
        sampled_records = random.sample(patient_records, min(self.data_per_patient, len(patient_records)))
        # print(f"patient_id: {patient_id}")
        # print(f"sampled_records: {sampled_records}")

        max_audio_len = 16000 * self.audio_duration

        data, text_data, speaker_id, phq_symptoms, phq_binary, len_data = [], [], [], [], [], []
        concatenated_audio = torch.empty(0)
        # sample record form each patient 
        for record in sampled_records:
            # Load audio data
            audio, _ = torchaudio.load(record[1])
            audio = audio[0]

            # # Pad or crop the audio

            if audio.size(0) > max_audio_len:
                audio = audio[:max_audio_len]               

            # Concatenate the audio to the main tensor
            concatenated_audio = torch.cat((concatenated_audio, audio), dim=0)

            # Check if the concatenated length is too long, and if so, crop it
            if concatenated_audio.size(0) > max_audio_len:
                concatenated_audio = concatenated_audio[:max_audio_len]
                break  # Stop further processing since we've reached the maximum length

            if self.is_train:
                audio = audio.detach().cpu().numpy()
                audio = self.transform(samples=audio, sample_rate=16000)
                audio = torch.tensor(audio)

            speaker_identifier = int(record[1].split('_')[-1][0]) 
            
            files = pd.read_csv(record[2])
            txt_data = files[files['index'] == speaker_identifier]['utterance'].values.tolist()
            
            print("read sample from: ", record[2])
            print("txt_data", txt_data)
            
            data.append(audio)
            text_data.append(txt_data)
            speaker_id.append(record[0])
            phq_symptoms.append(self._get_phq_symptoms(record[4]))
            phq_binary.append(torch.tensor(record[3]))
            len_data.append(torch.tensor(len(audio)))
            
        # After the loop, check if the final concatenated audio needs to be cropped
        if concatenated_audio.size(0) > max_audio_len:
            concatenated_audio = concatenated_audio[:max_audio_len]
        data =concatenated_audio 
        # Stack data for the patient
        # data = torch.stack(data, dim=0)
        len_data = torch.stack(len_data, dim=0)
        phq_symptoms = torch.stack(phq_symptoms, dim=0)
        phq_binary = torch.stack(phq_binary, dim=0)


        return data, text_data, speaker_id, phq_symptoms, phq_binary, len_data

    def _get_phq_symptoms(self, phq8_label):
        """
        Converts PHQ-8 symptom labels into binary format based on the selected 5 symptoms.
        """
        phq_nointerest = 1 if phq8_label[0] > 0 else 0
        phq_depressed = 1 if phq8_label[1] > 0 else 0
        phq_sleep = 1 if phq8_label[2] > 0 else 0
        phq_tired = 1 if phq8_label[3] > 0 else 0
        phq_failure = 1 if phq8_label[5] > 0 else 0

        # Create the binary symptom tensor
        phq_symptoms = torch.tensor([phq_nointerest, phq_depressed, phq_sleep, phq_tired, phq_failure]).float()
        return phq_symptoms

class DepressionDatasetGenerator(Dataset):
    def __init__(
        self,
        data_list:              list,
        noise_list:             list,
        data_len:               int,
        is_train:               bool=False,
        audio_duration:         int=6,
        model_type:             str="rnn",
        apply_guassian_noise:   bool=False,
        dataset:                   str='iemocap'
    ):
        """
        Set dataloader for depression finetuning.
        :param data_list:       Audio list files
        :param noise_list:      Audio list files
        :param data_len:        Length of input audio file size
        :param is_train:        Flag for dataloader, True for training; False for dev
        :param audio_duration:  Max length for the audio length
        :param model_type:      Type of the model
        """
        self.data_list              = data_list
        self.noise_list             = noise_list
        self.data_len               = data_len
        self.is_train               = is_train
        self.audio_duration         = audio_duration
        self.model_type             = model_type
        self.apply_guassian_noise   = apply_guassian_noise
        self.data                   = dataset 

        self.transform = Compose([
            AddGaussianSNR(min_snr_in_db=10.0, max_snr_in_db=30.0, p=1.0),
            TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
        ])
        
    def __len__(self):
        return self.data_len

    def __getitem__(
        self, item
    ):  
        # Read original speech in dev
        data, _ = torchaudio.load(self.data_list[item][1])
        
        ### extract text data 
        file_path = self.data_list[item][2]
        
        speaker_identifier = int(self.data_list[item][1].split('_')[-1][0]) 
        
        if self.data in ['daic']: # txt trainscript가 아닌, csv에서 불러옴. 
            files = pd.read_csv(file_path)
            txt_data = files[files['index'] == speaker_identifier]['utterance'].values.tolist()

        # phq binary (depression labels)
        phq_label = self.data_list[item][3]
    
        # phq8 labels (symptom labels)
        phq8_label = self.data_list[item][4]
        
        
        phq_nointerest = 1 if phq8_label[0] > 0 else 0
        phq_depressed = 1 if phq8_label[1] > 0 else 0
        phq_sleep = 1 if phq8_label[2] > 0 else 0
        phq_tired = 1 if phq8_label[3] > 0 else 0
        phq_appetite = 1 if phq8_label[4] > 0 else 0
        phq_failure = 1 if phq8_label[5] > 0 else 0
        phq_concentrating = 1 if phq8_label[6] > 0 else 0
        phq_moving = 1 if phq8_label[7] > 0 else 0

        # all 8 symptoms         
        # phq_symptoms = torch.tensor([phq_nointerest, phq_depressed, phq_sleep, phq_tired, 
        #                              phq_appetite, phq_failure, phq_concentrating, phq_moving]).float()
        
        # select 5 symptoms based on prior work
        # loss of interest, depressed mood, Sleeping habits, Tiredness, Feeling of failure
        phq_symptoms = torch.tensor([phq_nointerest, phq_depressed, phq_sleep, phq_tired, 
                                     phq_failure]).float()
                
        data = data[0]
        if data.isnan()[0].item(): data = torch.zeros(data.shape)
        if len(data) > self.audio_duration*16000: data = data[:self.audio_duration*16000]
        if self.is_train:
            data = data.detach().cpu().numpy()
            data = self.transform(samples=data, sample_rate=16000)
            data = torch.tensor(data)
        
        # print(data, txt_data, phq_symptoms, phq_label)
                
        if self.data in ['daic']:
            return data, txt_data, speaker_identifier, phq_symptoms, phq_label

    def _padding_cropping(
        self, input_wav, size
    ):
        if len(input_wav) > size:
            input_wav = input_wav[:size]
        elif len(input_wav) < size:
            input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
        return input_wav

def collate_fn(batch):
    # max of 6s of data
    max_audio_len = min(max([b[0].shape[0] for b in batch]), 16000*6)

    data, text_data, speaker_id, taregt, len_data = list(), list(), list(), list(), list()

    for idx in range(len(batch)):
        # append data
        data.append(padding_cropping(batch[idx][0], max_audio_len))
        
        text_data.append(batch[idx][1])
        speaker_id.append(batch[idx][2])
        
        # append len
        if len((batch[idx][0])) >= max_audio_len: len_data.append(torch.tensor(max_audio_len))
        else: len_data.append(torch.tensor(len((batch[idx][0]))))
        
        # append target
        taregt.append(torch.tensor(batch[idx][3]))
    
    data = torch.stack(data, dim=0)
    len_data = torch.stack(len_data, dim=0)
    target = torch.stack(taregt, dim=0)


    # print(f"dataloader: {data.shape, len_data.shape, target.shape}")
    return data, text_data, speaker_id, target, len_data

def padding_cropping(
    input_wav, size
):
    if len(input_wav) > size:
        input_wav = input_wav[:size]
    elif len(input_wav) < size:
        input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
    return input_wav

class EmotionDatasetGenerator(Dataset):
    def __init__(
        self,
        data_list:              list,
        noise_list:             list,
        data_len:               int,
        is_train:               bool=False,
        audio_duration:         int=6,
        model_type:             str="rnn",
        apply_guassian_noise:   bool=False,
        dataset:                   str='iemocap'
    ):
        """
        Set dataloader for emotion recognition finetuning.
        :param data_list:       Audio list files
        :param noise_list:      Audio list files
        :param data_len:        Length of input audio file size
        :param is_train:        Flag for dataloader, True for training; False for dev
        :param audio_duration:  Max length for the audio length
        :param model_type:      Type of the model
        """
        self.data_list              = data_list
        self.noise_list             = noise_list
        self.data_len               = data_len
        self.is_train               = is_train
        self.audio_duration         = audio_duration
        self.model_type             = model_type
        self.apply_guassian_noise   = apply_guassian_noise
        self.data                   = dataset 

        self.transform = Compose([
            AddGaussianSNR(min_snr_in_db=10.0, max_snr_in_db=30.0, p=1.0),
            TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
        ])
        
    def __len__(self):
        return self.data_len

    def __getitem__(
        self, item
    ):
        # Read original speech in dev
        data, _ = torchaudio.load(self.data_list[item][3])
        
        ### extract text data 
        file_path = self.data_list[item][4]
        
        speaker_identifier = self.data_list[item][0]csv에서 불러옴. 
            files = pd.read_csv(file_path)
            txt_data = files[files['Speaker'] == speaker_identifier]['Utterance'].values.tolist()
        elif self.data == 'meld':
            files = pd.read_csv(file_path)
            txt_data = files[files['Sr No.'] == speaker_identifier]['Utterance'].values.tolist()

        data = data[0]
        if data.isnan()[0].item(): data = torch.zeros(data.shape)
        if len(data) > self.audio_duration*16000: data = data[:self.audio_duration*16000]
        if self.is_train:
            data = data.detach().cpu().numpy()
            data = self.transform(samples=data, sample_rate=16000)
            data = torch.tensor(data)
        if self.data in ['iemocap','iemocap6']:
            return data, txt_data, int(self.data_list[item][5])-1, self.data_list[item][-1]
        else:
            return data, txt_data, int(self.data_list[item][5]), self.data_list[item][-1]

    def _padding_cropping(
        self, input_wav, size
    ):
        if len(input_wav) > size:
            input_wav = input_wav[:size]
        elif len(input_wav) < size:
            input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
        return input_wav
    

def include_for_finetune(
    data: list, dataset: str
):
    """
    Return flag for inlusion of finetune.
    :param data:        Input data entries [key, filepath, labels]
    :param dataset:     Input dataset name
    :return: flag:      True to include for finetuning, otherwise exclude for finetuning
    """
    if dataset in ["iemocap", "iemocap_impro"]:
        # IEMOCAP data include 4 emotions, exc->hap
        if data[-1] in ["neu", "sad","fru", "ang" , "exc", "hap"]: return True

    if dataset in ['iemocap6']:
        if data[-1] in ["neu", "sad", "fru", "ang" , "hap", "exc"]: return True
    if dataset == "meld":
        # MELD data include 4 emotions
        #if data[-1] in ["neutral", "sadness", "anger", "joy", "surprise", "fear", "disgust"]: return True
        if data[-1] in ["neutral", "sadness", "anger", "joy"]: return True
    if dataset == "meld6":
        if data[-1] in ["neutral", "sadness", "anger", "joy", "surprise", "fear", "disgust"]: return True
    if dataset == "cmu-mosei": return True
    if dataset == "ravdess": return True
    return False

def map_label(
    data: list, dataset: str
):  
    """
    Return labels for the input data.
    :param data:        Input data entries [key, filepath, labels]
    :param dataset:     Input dataset name
    :return label:      Label index: int
    """
    label_dict = {
        "iemocap6": {"neu": 0, "sad": 1, "fru": 2, "ang": 3,"hap": 4, "exc": 5},
        "iemocap": {"neu": 0, "sad": 1, "fru": 1, "ang": 2, "exc": 3, "hap": 3},
        "iemocap_impro": {"neu": 0, "sad": 1, "ang": 2, "exc": 3, "hap": 3},

        "meld": {"neutral": 0, "sadness": 1,"anger": 2, "joy": 3},
        "meld6": {"neutral": 0, "sadness": 1,  "anger": 2, "joy": 3, "surprise": 4, "fear":5, "disgust":6},
    }
    if dataset in ["iemocap", "iemocap6", "meld", "crema_d", "iemocap_impro"]:
        return label_dict[dataset][data[-1]]
    if dataset in ["cmu-mosei"]:
        # if data[-1] == 0: return 0
        if data[-1] > 0: return 0
        elif data[-1] <= 0: return 1
    if dataset in ["ravdess"]:
        # calm case, merge with neutral
        if data[-1] == 1: return data[-1]-1
        return data[-1]-2
        
def log_dataset_details(
    input_data_list:    list,
    split:              str,
    dataset:            str
):  
    """
    Log the label distribution of the dataset given the split.
    :param input_data_list:     Input data entries [key, filepath, labels]
    :param split:               Splits: train/dev/test
    :param dataset:             Input dataset name
    :return label_stats: stats of the datasets
    """
    label_dict = {
        "iemocap6": {0: "neu", 1: "sad", 2: "fru", 3: "ang", 4: "hap", 5: "exc"},
        "iemocap": {0: "neu", 1: "sad", 2: "ang", 3: "hap"},
        "iemocap_impro": {0: "neu", 1: "sad", 2: "ang", 3: "hap"},
    
        "meld6": {0: "neutral", 1: "sadness", 2: "anger", 3: "joy", 4: "surprise", 5: "fear", 6: "disgust"},
        "meld": {0: "neutral", 1: "sadness", 2: "anger", 3: "joy"},

        "cmu-mosei": {0: "postive", 1: "negative"},
        "ravdess": {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fearful", 5: "disgust", 6: "surprised"}
        
    }
    
    label_stats = dict()
    for data in input_data_list:
        # print("input_data_list", input_data_list)
        if data[-1] not in label_stats: label_stats[data[-1]] = 0
        label_stats[data[-1]] += 1
    
    logging.info(f'------------------------------------------------')
    logging.info(f'Number of {split} audio files {dataset}: {len(input_data_list)}')
    for label in label_stats:
        logging.info(f'Number of {split} audio files {label_dict[dataset][label]}: {label_stats[label]}')
    logging.info(f'------------------------------------------------')
    return label_stats
    

def load_pretrain_audios(
    input_path: str
):
    """
    Load pretrain audio data.
    :param input_path: Input data path
    :return train_file_list, dev_file_list: train and dev file list, we don't have test in pretrain
    """
    train_file_list, dev_file_list = list(), list()
    train_stats_dict, dev_stats_dict = dict(), dict()
    for dataset in ['iemocap', 'iemocap6', 'meld',  'ravdess',  'cmu-mosei', ]:
        with open(str(Path(input_path).joinpath(f'{dataset}.json')), "r") as f: 
            split_dict = json.load(f)
        # some stats
        train_stats_dict[dataset] = len(split_dict['train'])
        dev_stats_dict[dataset] = len(split_dict['dev'])

        for split in ['train', 'dev']:
            for data in split_dict[split]:
                if split == 'train': train_file_list.append(data)
                elif split == 'dev': dev_file_list.append(data)

    # logging train file nums
    logging.info(f'------------------------------------------------')
    logging.info(f'Number of train audio files {len(train_file_list)}')
    logging.info(f'------------------------------------------------')
    for dataset in ['iemocap','iemocap6', 'meld', 'meld6', 'ravdess', 'cmu-mosei']:
        logging.info(f'Number of train audio files {dataset}: {train_stats_dict[dataset]}')
    logging.info(f'------------------------------------------------')
    
    # logging dev file nums
    logging.info(f'------------------------------------------------')
    logging.info(f'Number of dev audio files {len(dev_file_list)}')
    logging.info(f'------------------------------------------------')
    for dataset in ['iemocap','iemocap6', 'meld', 'meld6', 'ravdess',  'cmu-mosei']:
        logging.info(f'Number of dev audio files {dataset}: {dev_stats_dict[dataset]}')
    logging.info(f'------------------------------------------------')
    
    return train_file_list, dev_file_list


def load_finetune_audios(
    input_path:     str,
    audio_path:     str,
    dataset:        str,
    fold_idx:       int
):
    """
    Load finetune audio data.
    :param input_path:  Input data path
    :param dataset:     Dataset name
    :param fold_idx:    Fold idx
    :return train_file_list, dev_file_list: train, dev, and test file list
    """
    train_file_list, dev_file_list, test_file_list = list(), list(), list()
    if dataset in ["iemocap_impro"]:
        with open(str(Path(input_path).joinpath(f'iemocap_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ["crema_d_complete"]:
        with open(str(Path(input_path).joinpath(f'crema_d_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ["iemocap", "iemocap6","crema_d", "ravdess", "msp-improv"]:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ["msp-podcast", "meld", "meld6"]: ## add meld
        with open(str(Path(input_path).joinpath(f'{dataset}.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ['daic']:
        with open(str(Path(input_path).joinpath(f'train_val_test_split.json')), "r") as f: split_dict = json.load(f)
    
    
    for split in ['train', 'dev', 'test']:
        for data in split_dict[split]:
            # pdb.set_trace()
            if include_for_finetune(data, dataset):
                data[-1] = map_label(data, dataset)
                if dataset == "iemocap_impro" and "impro" not in data[0]: continue
                speaker_id, file_path  = data[1], data[3]

                if dataset in ['iemocap', 'iemocap6','msp-improv','crema_d', 'msp-podcast']:
                    output_path = Path(audio_path).joinpath(dataset, file_path.split('/')[-1])
                
                elif dataset in ['ravdess', 'emov_db', 'vox-movie']:
                    output_path = Path(audio_path).joinpath(dataset, f'{speaker_id}_{file_path.split("/")[-1]}')
                #data[3] = str(output_path)
                if split == 'train': train_file_list.append(data)
                elif split == 'dev': dev_file_list.append(data)
                elif split == 'test': test_file_list.append(data)
            
            # DAIC 
            else: 
                if split == 'train': train_file_list.append(data)
                elif split == 'dev': dev_file_list.append(data)
                elif split == 'test': test_file_list.append(data)
    
    if dataset in ['daic']:         
        print(len(train_file_list), len(dev_file_list), len(test_file_list))
        
        if len(dev_file_list) == 0: return train_file_list, test_file_list 
        return train_file_list, dev_file_list, test_file_list    


    # logging train/dev/test file nums
    log_dataset_details(train_file_list, split='train', dataset=dataset)
    log_dataset_details(dev_file_list, split='dev', dataset=dataset)
    log_dataset_details(test_file_list, split='test', dataset=dataset)
    return train_file_list, dev_file_list, test_file_list


def count_class_instances(input_file_list):
    """
    Count the number of instances for each class in the dataset.
    :param input_file_list:        Input data entries [key, filepath, labels]
    :return class_counts:          Dictionary with counts for each class
    """
    class_counts = {0: 0, 1: 0}

    for data in input_file_list:
        label = data[3]  # Assuming data[3] contains the binary label (0 or 1)
        class_counts[label] += 1

    return class_counts


def return_weights(
    input_path:     str,
    dataset:        str,
    fold_idx:       int
):
    """
    Return training weights.
    :param input_path:  Input data path
    :param dataset:     Dataset name
    :param fold_idx:    Fold idx
    :return weights:    Class weights
    """
    train_file_list = list()
    if dataset in ["iemocap_impro"]:
        with open(str(Path(input_path).joinpath(f'iemocap_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    elif dataset in ["crema_d_complete"]:
        with open(str(Path(input_path).joinpath(f'crema_d_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    elif dataset in ["msp-podcast", "meld"]:
        with open(str(Path(input_path).joinpath(f'{dataset}.json')), "r") as f:
            split_dict = json.load(f) 
    else:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    
    for data in split_dict['train']:
        if include_for_finetune(data, dataset):
            data[-1] = map_label(data, dataset)
            train_file_list.append(data)
            
    # logging train file nums
    weights_stats = log_dataset_details(train_file_list, split='train', dataset=dataset)
    # compute weight 
    weights = torch.tensor([weights_stats[c] for c in range(len(weights_stats))]).float()
    weights = weights.sum() / weights
    weights = weights / weights.sum()

    return weights

def return_dataset_stats(
    input_path:     str,
    dataset:        str,
    fold_idx:       int
):
    """
    Return training weights.
    :param input_path:  Input data path
    :param dataset:     Dataset name
    :param fold_idx:    Fold idx
    :return weights:    Class weights
    """
    train_file_list = list()
    if dataset in ["iemocap_impro"]:
        with open(str(Path(input_path).joinpath(f'iemocap_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    elif dataset in ["crema_d_complete"]:
        with open(str(Path(input_path).joinpath(f'crema_d_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    elif dataset in ["msp-podcast"]:
        with open(str(Path(input_path).joinpath(f'{dataset}.json')), "r") as f:
            split_dict = json.load(f)
    else:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    
    for split in ["train", "dev", "test"]:
        for data in split_dict[split]:
            if include_for_finetune(data, dataset):
                data[-1] = map_label(data, dataset)
                train_file_list.append(data)
            
    # logging train file nums
    log_dataset_details(train_file_list, split='train', dataset=dataset)

def return_speakers(
    input_file_list:    list
):
    """
    Return training weights.
    :param input_file_list:     input file list
    :return speakers:           unique speakers
    """
    speaker_list = list()
    for input_data in input_file_list: speaker_list.append(input_data[1])
    speaker_list = list(set(speaker_list))
    speaker_list.sort()
    return speaker_list

def set_finetune_dataloader(
    args:                   dict,
    input_file_list:        list,
    is_train:               bool,
    is_distributed:         bool=False,
    rank:                   int=0,
    world_size:             int=2,
    apply_guassian_noise:   bool=False,
    
    # depression 
    symptom_class_weight:   float=None, 
    label_class_weight:     float=None
):
    """
    Return dataloader for finetune experiments.
    :param data:                    Input data entries [key, filepath, labels]
    :param is_train:                Flag for training or not
    :param is_distributed:          Flag for distributed training or not
    :param rank:                    Current GPU rank
    :param world_size:              Total GPU sizes
    :param apply_guassian_noise:    Apply Guassian Noise to audio or not
    :return dataloader:             Dataloader
    """

    # noise files
    noise_wav_files = glob.glob(
        "/media/data/projects/speech-privacy/emo2vec/noise_audio/*.wav"
    )
    
    # dataloader
    filtered_file_list = list()
    for file_path in input_file_list:
        if file_path[3] not in [
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/-7161_hlBOP5NskhM.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/678639_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/607281_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/730042_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/643200_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/570565_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/78720_ULkFbie8g-I.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/0_z7FicxE_pMU.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/7524_-mJ2ud6oKI8.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/78403_P0WaXnH37uI.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/255120_ULkFbie8g-I.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/0_278474.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/77605_bUFAN2TgPaU.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/491385_9bAgEmihzLs.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/96761_-mJ2ud6oKI8.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/133159_TxRS6vJ9ak0.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/768515_9K5mYSaoBL4.wav"
        ]:
            filtered_file_list.append(file_path)
            
    if args.dataset == 'daic' and symptom_class_weight is None or label_class_weight is None:
        
        class_counts = count_class_instances(filtered_file_list)
        num_negative = class_counts[0]
        num_positive = class_counts[1]

        # Compute positive weight
        pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32).to('cuda')

        # Store pos_weight in args for later use
        args.pos_weight = pos_weight
            
    if args.dataset == 'daic':
        data_generator = PatientDepressionDatasetGenerator(
            data_list=filtered_file_list, 
            noise_list=noise_wav_files,
            data_len=len(filtered_file_list),
            is_train=is_train,
            audio_duration = args.max_audio_len,
            model_type=args.downstream_model,
            apply_guassian_noise=apply_guassian_noise,
            dataset=args.dataset,
            data_per_patient=args.data_per_patient
        )

        if is_distributed:
            datasampler = torch.utils.data.distributed.DistributedSampler(
                data_generator, shuffle=True
            )

            dataloader = DataLoader(
                data_generator, 
                batch_size=args.batch_size, 
                num_workers=2, 
                drop_last=True,
                sampler=datasampler
            )
        else:
            if is_train:
                dataloader = DataLoader(
                    data_generator, 
                    batch_size=args.batch_size, 
                    num_workers=6, 
                    shuffle=is_train,
                    collate_fn=collate_fn3,
                    drop_last=is_train
                )
            else:
                dataloader = DataLoader(
                    data_generator, 
                    batch_size= 11, #args.batch_size, 
                    num_workers=6, 
                    shuffle=is_train,
                    collate_fn=collate_fn3,
                    drop_last=is_train
                )
    
    else:
        data_generator = EmotionDatasetGenerator(
            data_list=filtered_file_list, 
            noise_list=noise_wav_files,
            data_len=len(filtered_file_list),
            is_train=is_train,
            audio_duration = args.max_audio_len,
            model_type=args.downstream_model,
            apply_guassian_noise=apply_guassian_noise,
            dataset=args.dataset
        )

        if is_distributed:
            datasampler = torch.utils.data.distributed.DistributedSampler(
                data_generator, shuffle=True
            )

            dataloader = DataLoader(
                data_generator, 
                batch_size=32, 
                num_workers=2, 
                drop_last=True,
                sampler=datasampler
            )
        else:
            if is_train:
                dataloader = DataLoader(
                    data_generator, 
                    batch_size=32, 
                    num_workers=6, 
                    shuffle=is_train,
                    collate_fn=collate_fn,
                    drop_last=is_train
                )
            else:
                dataloader = DataLoader(
                    data_generator, 
                    batch_size=32, 
                    num_workers=6, 
                    shuffle=is_train,
                    collate_fn=collate_fn,
                    drop_last=is_train
                )
    return dataloader
