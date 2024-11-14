import pandas as pd
import os
import wave
import contextlib
import numpy as np
from pydub import AudioSegment

'''
NOTE: [Last Update] 0930 
audio file 전처리 (발화 단위로 cutting).
기존 데이터 전처리가 Ellie 발화를 포함하고 있어, 
환자 데이터만 추출 후, 한 발화씩 데이터 crop. 
''' 
    
# 첫 질문 이전 Ellie 발화 (Ellie 혼자 설명하는 거 다 삭제)
# 458 Ellie 발화 누락
# 451 Ellie 발화 누락
# 307 where are you from originally
# 308 where are you from originally
# 480 Ellie 발화 누락

def get_wav_time(wav_path):
    with contextlib.closing(wave.open(wav_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

def get_precise_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    '''
    Slicing audio from start_time to end_time with precise floating-point control.
    We use floats to represent time down to very small units (microseconds).
    '''
    # Convert to milliseconds as floats to keep precision
    start_time_ms = start_time * 1000  # Convert to milliseconds
    end_time_ms = end_time * 1000  # Convert to milliseconds

    sound = AudioSegment.from_wav(main_wav_path)  # Use .from_wav for wav files
    
    # Adjust the slicing for very small durations using floats
    if end_time_ms > start_time_ms:
        word = sound[start_time_ms:end_time_ms]
        word.export(part_wav_path, format="wav")
    else:
        print(f"Skipping slice with invalid duration: start_time = {start_time}, end_time = {end_time}")

def process_wav_files_for_all_patients(root_dir, output_base_dir):
    # Ensure output_base_dir exists
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Get all patient directories (e.g., 300_P, 301_P, etc.)
    for patient_dir in os.listdir(root_dir):
        full_patient_dir = os.path.join(root_dir, patient_dir)

        if os.path.isdir(full_patient_dir) and patient_dir.endswith('_P'):
            patient_id = patient_dir.split('_')[0]
            print(f"Processing patient {patient_id}...")

            # Set paths for the audio and transcript files
            audio_path = os.path.join(full_patient_dir, f'{patient_id}_AUDIO.wav')
            transcript_path = os.path.join(full_patient_dir, f'{patient_id}_TRANSCRIPT.csv')

            output_dir = os.path.join(output_base_dir, f'{patient_id}_processed')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)  # Create the directory if it doesn't exist

            # Check if the audio and transcript files exist
            if os.path.exists(audio_path) and os.path.exists(transcript_path):
                df = pd.read_csv(transcript_path, sep='\t')

                # First, look for "how are you doing today"
                ellie_question_time = df[(df['speaker'] == 'Ellie') & (df['value'].str.contains("how are you doing today", case=False))]

                # If "how are you doing today" is not found, look for "where are you from originally"
                if ellie_question_time.empty:
                    ellie_question_time = df[(df['speaker'] == 'Ellie') & (df['value'].str.contains("where are you from originally", case=False))]

                # If any of the questions are found, process the participant's response
                if not ellie_question_time.empty:
                    question_end_time = ellie_question_time.iloc[0]['stop_time']

                    # Now extract only participant's speech after this point
                    df_participant = df[(df['speaker'] == 'Participant') & (df['start_time'] >= question_end_time)]

                    index = 1
                    for _, row in df_participant.iterrows():
                        start_time = float(row['start_time'])
                        end_time = float(row['stop_time'])

                        # Skip if the time difference is too small or invalid
                        if end_time - start_time <= 0:
                            print(f"Skipping invalid segment with start_time: {start_time} and end_time: {end_time} for patient {patient_id}")
                            continue

                        output_wav = os.path.join(output_dir, f"{patient_id}_{index}.wav")

                        # Slice the audio to get participant's response with precise time slicing
                        get_precise_part_wav(audio_path, start_time, end_time, output_wav)
                        index += 1
                else:
                    print(f"Neither 'how are you doing today' nor 'where are you from originally' found for patient {patient_id}")
            else:
                print(f"Either audio or transcript file not found in {full_patient_dir}")

if __name__ == '__main__':
    # Root directory where all patient folders are located
    root_dir = '/home/dilab/hrlee/test/daic_woz_process/DAIC'
    # Base output directory for all processed files
    output_base_dir = '/home/dilab/hrlee/test/daic_woz_process/DAIC_processed'

    process_wav_files_for_all_patients(root_dir, output_base_dir)
