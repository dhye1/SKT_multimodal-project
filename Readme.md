## SKT AI Fellowship - Multimodal Project (2024.05 - 2024.11)
### Multimodal Emotion Recognition and Depression Detection
This repository contains a model for emotion recognition and depression detection using a multimodal dataset (text and audio). 

Our demo code is available at [https://github.com/dhye1/SKT_Chatbot_Demo ](https://github.com/dhye1/SKT_Chatbot_Demo) .

## 1. Emotion Recognition

### Data

The experiment uses the multimodal emotion dataset [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm). Download the dataset from the link provided and place it in the ```./data``` directory. Each file (IEMOCAP_{train/dev/test}.csv) contains text data and corresponding emotion labels for each utterance.


```
+--data
  +--IEMOCAP_train.csv
  +--IEMOCAP_dev.csv
  +--IEMOCAP_test.csv

  # Downloaded from IEMOCAP
  +--Session1
  +--Session2
  +--Session3
  +--Session4
  +--Session5

```
### Model Run
To train the model, execute:
```
cd bash
bash run_emotion.sh
```

### File Description
> ```experiment/finetune_emotion.py``` The training script for the emotion recognition model
> 
> ```model/custom_roberta.py``` Text processing model based on RoBERTa, with an added cross-modal attention layer. 
> 
> ```model/whisper_model.py``` Audio processing model based on Whisper.
> 
> ```prediction.py``` Class prediction layer
> 
> ```train_split/iemocap6.json``` Directory containing the training dataset. 
>
> ```train_split_gen/iemocap6.py``` Update the paths in the ```iemocap6.json``` to algin your directory structure. Ensure that the directory has the appropriate ```.wav``` and ```.csv``` file directory. 

```
    "train": [
        [
            "Ses01F_impro01_F000",
            "iemocap_Ses01F",
            "female",
            "{your_directory}/data/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav", # audio (.wav) file directory
            "{your_directory}/data/IEMOCAP_train.csv", # .csv file with text data
            "1",
            "neu"
        ],
```
> 





## 2. Depression Detection 

### Data
For Depression Detection, you need to download the **[DAIC_Woz Dataset](https://dcapswoz.ict.usc.edu)** dataset. Before running the model, it is essential to download and preprocess the required data. Detailed instructions for data preprocessing can be found in the **preprocessing** folder.

### Model Run
The main script to run the model is located at: `bash/run_depression_patient.sh`.

This script uses a pretrained emotion model and runs the depression prediction by executing the following Python file: `experiment/finetune_depression_symweight_simple_patient.py`.

### File Description
> `experiment/finetune_depression_symweight_simple_patient.py`: The training script for the depression detection model.
> 
> `model/custom_roberta_p.py`: Text processing model based on RoBERTa, with an added cross-modal attention layer for depression detection.
> 
> `model/depression_prediction.py`: Class prediction layer including symptom and depression classifier.
> 
> `dataloader/dataloader_dep.py`: Data loader for the depression model, loading patient-wise data.
