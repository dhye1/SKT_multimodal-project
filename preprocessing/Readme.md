## Overview
This directory contains scripts for preprocessing audio and text data for model training. It includes the following scripts:

1. **Preprocessing_audio_text.py**: Generates a JSON file for training using preprocessed audio and text files.
2. **Preprocessing_Cutting_custom.py**: Splits audio files into segments based on utterances.
3. **Preprocessing_Merging_custom.py**: Merges text and audio files into the specified number of utterances.

To execute these scripts, you need to download the **[DAIC_Woz Dataset](https://dcapswoz.ict.usc.edu)** dataset.

## Execution Order
To ensure proper data processing, execute the scripts in the following order:

1. **Preprocessing_Cutting_custom.py**
   - Splits the audio files into utterance-level segments.
2. **Preprocessing_Merging_custom.py**
   - Merges the segmented audio files and corresponding text into specified utterance blocks.
3. **Preprocessing_audio_text.py**
   - Uses the merged files to create a `train_val_test_split.json` file for model training 

## Output
After completing all preprocessing steps, the `train_val_test_split.json` file will be generated in the data directory.
