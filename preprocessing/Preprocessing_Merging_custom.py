import os
import re
import pandas as pd
from pydub import AudioSegment
from transformers import RobertaTokenizer

'''
tokenizer max txt len 기준으로 데이터 전처리
'''


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

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

def tokenize_texts(texts, tokenizer, max_txt_len=256, device='cuda'):
    flat_txt = flat_text(texts)
    encodings = tokenizer(flat_txt, padding=True, truncation=True, max_length=max_txt_len, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    return input_ids, attention_mask

def merging_text(num_utt, max_txt_len, root_dir, output_dir):
    '''
    Merges participant text utterances based on the number of utterances (num_utt).
    It handles both Participant and Ellie's utterances.
    This function processes all participants in the directory.

    :param num_utt: Number of participant utterances to merge.
    :param max_txt_len: Maximum token length for the merged text.
    '''
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all participant directories
    for participant_dir in os.listdir(root_dir):
        if not participant_dir.endswith('_P'):
            continue  # Skip any files or directories that do not match the participant pattern

        participant_id = participant_dir.split('_')[0]
        transcript_path = os.path.join(root_dir, participant_dir, f'{participant_id}_TRANSCRIPT.csv')

        if not os.path.exists(transcript_path):
            print(f"Transcript file not found for Participant {participant_id}")
            continue

        df = pd.read_csv(transcript_path, sep='\t')

        # Start from the utterance "how are you doing today" or "where are you from originally"
        ellie_question_time = df[(df['speaker'] == 'Ellie') & (df['value'].str.contains("how are you doing today", case=False))]

        if ellie_question_time.empty:
            ellie_question_time = df[(df['speaker'] == 'Ellie') & (df['value'].str.contains("where are you from originally", case=False))]

        if ellie_question_time.empty:
            print(f"Neither 'how are you doing today' nor 'where are you from originally' found for participant {participant_id}")
            continue

        question_end_time = ellie_question_time.iloc[0]['start_time']

        # Filter to get all utterances (Participant and Ellie) after the initial question
        df_filtered = df[(df['start_time'] >= question_end_time)]

        # Filter to end before the utterance "i think i've asked everything"
        ending_utterance = df[(df['speaker'] == 'Ellie') & (df['value'].str.contains("i think i've asked everything", case=False))]

        if not ending_utterance.empty:
            ending_time = ending_utterance.iloc[0]['start_time']
            df_filtered = df_filtered[df_filtered['start_time'] < ending_time]

        merged_utterances = []
        current_text = ""
        utt_count = 0
        prev_speaker = None
        utterances_buffer = []

        for idx, row in df_filtered.iterrows():
            speaker = row['speaker']
            utterance = row['value']

            # Add participant ID to Participant speaker
            if speaker == 'Participant':
                speaker = f"Participant {participant_id}"

            # Append utterance to buffer
            if speaker != prev_speaker or current_text == "":
                if current_text:
                    current_text += " </s></s> "
                current_text += f"{speaker}: {utterance}"
            else:
                current_text += f" {utterance}"

            utterances_buffer.append(row)

            # Count Participant's utterances
            if speaker.startswith('Participant'):
                utt_count += 1

            # When reaching num_utt or exceeding max length, finalize the current merged utterance
            current_tokenized_len = len(tokenizer(current_text, truncation=True, max_length=max_txt_len)['input_ids'])
            if utt_count == num_utt or current_tokenized_len > max_txt_len:
                # If last speaker is Ellie, stop before adding her last utterance
                if speaker.startswith('Ellie'):
                    merged_utterances.append((utterances_buffer[:-1], current_text.rsplit("</s></s>", 1)[0].strip()))
                else:
                    merged_utterances.append((utterances_buffer, current_text.strip()))
                current_text = ""
                utt_count = 0
                utterances_buffer = []

            prev_speaker = speaker

        # Save the last merged utterance if any exists
        if current_text:
            merged_utterances.append((utterances_buffer, current_text.strip()))

        # Saving the merged transcript data to a CSV file
        output_file = os.path.join(output_dir, f'{participant_id}_merge.csv')
        merged_df = pd.DataFrame({'index': range(1, len(merged_utterances) + 1), 'utterance': [x[1] for x in merged_utterances]})
        merged_df.to_csv(output_file, index=False)
        print(f"Text merging completed for {participant_id}.")

    return merged_utterances


def numerical_sort(value):
    """
    Sort key that extracts numbers from the filenames (e.g., '300_1.wav' -> 1).
    """
    parts = re.findall(r'\d+', value)
    return list(map(int, parts))  # Return numbers as integers for proper numerical sorting


def merging_audio(merged_text_data, root_dir, output_dir):
    '''
    Function to merge participant audio wav files based on the merged text data.
    This function processes all participants in the directory.

    :param merged_text_data: List of merged utterance data from the text merging step.
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all participant directories
    for participant_dir in os.listdir(root_dir):
        if not participant_dir.endswith('_processed'):
            continue  # Skip directories that are not processed participant directories

        participant_id = participant_dir.split('_')[0]
        participant_audio_dir = os.path.join(root_dir, participant_dir)

        if not os.path.exists(participant_audio_dir):
            print(f"Audio directory not found for Participant {participant_id}")
            continue

        # Sort audio files based on numbers in filenames
        audio_files = sorted([f for f in os.listdir(participant_audio_dir) if f.endswith('.wav')], key=numerical_sort)

        merged_count = 1
        for utterance_data, _ in merged_text_data:
            merged_audio = None

            # Merge audio files corresponding to the current merged text utterance
            for row in utterance_data:
                audio_index = row.name + 1
                audio_filename = f"{participant_id}_{audio_index}.wav"
                file_path = os.path.join(participant_audio_dir, audio_filename)

                if not os.path.exists(file_path):
                    print(f"Audio file not found: {file_path}")
                    continue

                sound = AudioSegment.from_wav(file_path)

                if merged_audio is None:
                    merged_audio = sound
                else:
                    merged_audio += sound

            # Save the merged result
            if merged_audio is not None:
                output_file = os.path.join(output_dir, f"{participant_id}_merge_{merged_count}.wav")
                merged_audio.export(output_file, format="wav")
                print(f"Merged file saved: {output_file}")
                merged_count += 1

        print(f"Audio merging completed for {participant_id}.")


if __name__ == '__main__':
    num_utt = 5  # Number of participant utterances to merge
    max_txt_len = 256  # Maximum text length for merged utterances

    root_dir_audio = '/home/dilab/hrlee/test/daic_woz_process/DAIC_processed'  # Root directory containing all processed audio
    output_dir_audio = f'/home/dilab/hrlee/test/daic_woz_process/DAIC_processed_merged/audio_maxx_{num_utt}'

    root_dir_txt = '/home/dilab/hrlee/test/daic_woz_process/DAIC'  # Root directory containing all participants
    output_dir_txt = f'/home/dilab/hrlee/test/daic_woz_process/DAIC_processed_merged/text_maxx_{num_utt}'
    
    merged_text_data = merging_text(num_utt, max_txt_len, root_dir_txt, output_dir_txt)  # Text merging for all participants
    merging_audio(merged_text_data, root_dir_audio, output_dir_audio)  # Audio merging for all participants

    if not os.path.exists(output_dir_audio):
        os.makedirs(output_dir_audio)

    if not os.path.exists(output_dir_txt):
        os.makedirs(output_dir_txt)