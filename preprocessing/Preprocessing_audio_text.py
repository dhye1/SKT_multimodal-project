import os
import json
import pandas as pd
import random
from collections import Counter, defaultdict
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

'''
학습을 위한 json파일 생성 

train_list의 participant_id 개수: (107 -> 101) / total instance: 3220
318, 321, 341, 362 (Out of sync) 319, 409 (labeling error) 제외  / total instance: 1254

val_list의 participant_id 개수: (35 -> 33)
451, 458 제외 (NO ELLIE)
'''


phq8_question_names = [
    'PHQ8_NoInterest',
    'PHQ8_Depressed',
    'PHQ8_Sleep',
    'PHQ8_Tired',
    'PHQ8_Appetite',
    'PHQ8_Failure',
    'PHQ8_Concentrating',
    'PHQ8_Moving'
]

# 수정된 plot_combined_counts 함수
def plot_combined_counts(train_counts, val_counts, test_counts, output_dir):
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 존재하는 데이터셋만을 포함하도록 datasets 딕셔너리 생성
    datasets = {
        'Train': train_counts,
        'Test': test_counts
    }
    if val_counts is not None:
        datasets['Validation'] = val_counts

    num_datasets = len(datasets)
    width = 0.8 / num_datasets  # 막대 너비를 데이터셋 수에 따라 조정

    # 먼저 PHQ8_Binary에 대한 그래프를 그립니다.
    question_name = 'PHQ8_Binary'
    class_labels = set()
    dataset_counts = {}
    dataset_participants = {}
    for dataset_name, counts_data in datasets.items():
        counts = counts_data['phq8_binary_counts']
        participants = counts_data['phq8_binary_participants']
        
        dataset_counts[dataset_name] = counts
        dataset_participants[dataset_name] = participants
        class_labels.update(counts.keys())

    class_labels = sorted(class_labels)
    x = np.arange(len(class_labels))  # 클래스 레이블 위치

    fig, ax = plt.subplots(figsize=(10, 6))

    # 각 데이터셋에 대한 막대 위치 설정
    offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, num_datasets)
    bars = {}
    for idx, (dataset_name, offset) in enumerate(zip(datasets.keys(), offsets)):
        counts = dataset_counts[dataset_name]
        participants = dataset_participants[dataset_name]
        counts_values = [counts.get(cls, 0) for cls in class_labels]
        participants_values = [participants.get(cls, 0) for cls in class_labels]

        positions = x + offset
        bars[dataset_name] = ax.bar(positions, counts_values, width, label=f'{dataset_name} Data Points')

        # 막대 위에 데이터 포인트 수와 참가자 수 표시
        for i, bar in enumerate(bars[dataset_name]):
            height = bar.get_height()
            participant = participants_values[i]
            ax.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)} ({participant})', ha='center', va='bottom', fontsize=9)

    # 그래프 설정
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number')
    ax.set_title(f'{question_name} Class Counts Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {cls}' for cls in class_labels])
    ax.legend()

    fig.tight_layout()
    # 그래프 저장
    plot_filename = f'{question_name}_class_counts.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    # 이제 나머지 PHQ8 질문들에 대한 그래프를 그립니다.
    for question_name in phq8_question_names:
        # 각 데이터셋에서 해당 질문의 클래스별 카운트 수집
        class_labels = set()
        dataset_counts = {}
        dataset_participants = {}
        for dataset_name, counts_data in datasets.items():
            counts = counts_data['phq8_labels_counts'][question_name]
            participants = counts_data['phq8_labels_participants'][question_name]
            
            dataset_counts[dataset_name] = counts
            dataset_participants[dataset_name] = participants
            class_labels.update(counts.keys())

        class_labels = sorted(class_labels)
        x = np.arange(len(class_labels))  # 클래스 레이블 위치

        fig, ax = plt.subplots(figsize=(10, 6))

        # 각 데이터셋에 대한 막대 위치 설정
        offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, num_datasets)
        bars = {}
        for idx, (dataset_name, offset) in enumerate(zip(datasets.keys(), offsets)):
            counts = dataset_counts[dataset_name]
            participants = dataset_participants[dataset_name]
            counts_values = [counts.get(cls, 0) for cls in class_labels]
            participants_values = [participants.get(cls, 0) for cls in class_labels]

            positions = x + offset
            bars[dataset_name] = ax.bar(positions, counts_values, width, label=f'{dataset_name} Data Points')

            # 막대 위에 데이터 포인트 수와 참가자 수 표시
            for i, bar in enumerate(bars[dataset_name]):
                height = bar.get_height()
                participant = participants_values[i]
                ax.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)} ({participant})', ha='center', va='bottom', fontsize=9)

        # 그래프 설정
        ax.set_xlabel('Classes')
        ax.set_ylabel('Number')
        ax.set_title(f'{question_name} Class Counts Across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {cls}' for cls in class_labels])
        ax.legend()

        fig.tight_layout()
        # 그래프 저장
        plot_filename = f'{question_name}_class_counts.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        
# 막대 그래프를 그리는 함수 정의
def plot_counts(counts_data, dataset_name, output_dir='plots'):
    
    # PHQ8_Binary 그래프
    phq8_binary_counts = counts_data['phq8_binary_counts']
    phq8_binary_participants = counts_data['phq8_binary_participants']

    classes = sorted(phq8_binary_counts.keys())
    counts = [phq8_binary_counts[cls] for cls in classes]
    participants = [phq8_binary_participants[cls] for cls in classes]

    x = range(len(classes))
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, counts, color='skyblue', label='Data Points')
    plt.plot(x, participants, color='orange', marker='o', label='Participants')
    plt.xticks(x, [f'Class {cls}' for cls in classes])
    plt.xlabel('Classes')
    plt.ylabel('Number')
    plt.title(f'{dataset_name} - PHQ8_Binary class counts')
    plt.legend()

    # 막대 위에 데이터 포인트 수와 참가자 수 표시
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        participant = participants[idx]
        plt.text(bar.get_x() + bar.get_width()/2, height,
                    f'{int(height)} ({participant})', ha='center', va='bottom')

    plt.tight_layout()
    # 그래프 저장
    binary_plot_path = os.path.join(output_dir, f'{dataset_name}_PHQ8_Binary.png')
    plt.savefig(binary_plot_path)
    plt.close()

    # PHQ8_Label 그래프
    for question_name in phq8_question_names:
        counts = counts_data['phq8_labels_counts'][question_name]
        participants = counts_data['phq8_labels_participants'][question_name]

        scores = sorted(counts.keys())
        count_values = [counts[score] for score in scores]
        participant_values = [participants[score] for score in scores]

        x = range(len(scores))
        plt.figure(figsize=(8, 6))
        bars = plt.bar(x, count_values, color='skyblue', label='Data Points')
        plt.plot(x, participant_values, color='orange', marker='o', label='Participants')
        plt.xticks(x, [f'Class {score}' for score in scores])
        plt.xlabel('Classes')
        plt.ylabel('Number')
        plt.title(f'{dataset_name} - {question_name} class counts')
        plt.legend()

        # 막대 위에 데이터 포인트 수와 참가자 수 표시
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            participant = participant_values[idx]
            plt.text(bar.get_x() + bar.get_width()/2, height,
                        f'{int(height)} ({participant})', ha='center', va='bottom')

        plt.tight_layout()
        # 그래프 저장
        label_plot_path = os.path.join(output_dir, f'{dataset_name}_{question_name}.png')
        plt.savefig(label_plot_path)
        plt.close()


def count_classes(data_list):
    # 데이터 포인트 수와 participant_id 수를 저장하기 위한 딕셔너리
    phq8_binary_counts = Counter()
    phq8_binary_participants = defaultdict(set)

    phq8_labels_counts = {q: Counter() for q in phq8_question_names}
    phq8_labels_participants = {q: defaultdict(set) for q in phq8_question_names}

    for item in data_list:
        participant_id = item[0]
        phq8_binary = item[3]
        phq8_labels = item[4]

        phq8_binary_counts[phq8_binary] += 1
        phq8_binary_participants[phq8_binary].add(participant_id)

        for idx, score in enumerate(phq8_labels):
            question_name = phq8_question_names[idx]
            phq8_labels_counts[question_name][score] += 1
            phq8_labels_participants[question_name][score].add(participant_id)

    # 결과를 딕셔너리로 반환
    result = {
        'phq8_binary_counts': phq8_binary_counts,
        'phq8_binary_participants': {k: len(v) for k, v in phq8_binary_participants.items()},
        'phq8_labels_counts': phq8_labels_counts,
        'phq8_labels_participants': {
            q: {score: len(ids) for score, ids in scores.items()}
            for q, scores in phq8_labels_participants.items()
        }
    }
    return result

# 클래스별 개수 출력 함수 정의
def print_counts(counts_data, dataset_name):
    phq8_binary_counts = counts_data['phq8_binary_counts']
    phq8_binary_participants = counts_data['phq8_binary_participants']

    print(f"\n{dataset_name} - PHQ8_Binary class counts:")
    for cls in sorted(phq8_binary_counts.keys()):
        count = phq8_binary_counts[cls]
        participants = phq8_binary_participants[cls]
        print(f"  Class {cls}: {count} ({participants})")

    phq8_labels_counts = counts_data['phq8_labels_counts']
    phq8_labels_participants = counts_data['phq8_labels_participants']

    for question_name in phq8_question_names:
        counts = phq8_labels_counts[question_name]
        participants = phq8_labels_participants[question_name]
        print(f"{dataset_name} - {question_name} class counts:")
        for score in sorted(counts.keys()):
            count = counts[score]
            num_participants = participants[score]
            print(f"  Score {score}: {count} ({num_participants})")


def preprocessing_data(audio_dir, txt_dir, label_file):
    """
    전처리한 audio / text파일을 활용하여 train / test / validation용 JSON 파일을 생성하여 저장하는 함수.
    
    participant_id가 318, 321, 341, 362, 409인 경우 제외
    
    :param audio_dir: 병합된 오디오 파일들이 저장된 경로
    :param txt_dir: 병합된 텍스트 파일들이 저장된 경로
    :param label_file: PHQ8 관련 라벨이 저장된 CSV 파일 경로
    :return: 모든 데이터 리스트 (각 참가자별 wav, txt, PHQ8 라벨 정보 포함)
    """
    
    # 제외할 participant_id 목록
    # 318, 321, 341, 362 : out of sync
    # 409 : labelling error - whose PHQ-8 score was 10 but the binary value given was 0 rather than 1.
    
    exclude_ids = [318, 321, 341, 362, 409]
    
    # Load the label file
    label_df = pd.read_csv(label_file)
    label_df = label_df.dropna()
    
    # 전처리된 audio, txt 파일 순서대로 로드
    all_data = []
    
    # 오디오 파일 이름에서 Participant_ID를 추출하고 해당 오디오/텍스트 파일 쌍을 처리
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith('.wav'):
            # Participant ID 추출 (예: "300_merge_1.wav" -> 300)
            participant_id = int(audio_file.split('_')[0])
            
                        
            # 제외할 participant_id인 경우 건너뜀
            if participant_id in exclude_ids:
                print(f"Skipping Participant ID {participant_id}")
                continue
            
            # 텍스트 파일 경로 및 해당 ID의 라벨 가져오기
            txt_file = os.path.join(txt_dir, f'{participant_id}_merge.csv')
            
            # 텍스트 파일 및 라벨 존재 여부 확인
            if not os.path.exists(txt_file):
                print(f"Text file not found for Participant {participant_id}")
                continue

            # Participant_ID에 해당하는 라벨 정보 가져오기
            label_row = label_df[label_df['Participant_ID'] == participant_id]
            
            # 라벨이 없는 경우 처리
            if label_row.empty:
                print(f"No label found for Participant ID {participant_id}")
                continue
            
            phq8_binary = int(label_row.iloc[0]['PHQ8_Binary'])  # int 변환
            phq8_labels = [
                int(label_row.iloc[0]['PHQ8_NoInterest']),
                int(label_row.iloc[0]['PHQ8_Depressed']),
                int(label_row.iloc[0]['PHQ8_Sleep']),
                int(label_row.iloc[0]['PHQ8_Tired']),
                int(label_row.iloc[0]['PHQ8_Appetite']),
                int(label_row.iloc[0]['PHQ8_Failure']),
                int(label_row.iloc[0]['PHQ8_Concentrating']),
                int(label_row.iloc[0]['PHQ8_Moving'])
            ]

            # audio / txt 쌍과 라벨을 리스트로 저장
            all_data.append([
                participant_id,
                os.path.join(audio_dir, audio_file),
                txt_file,
                phq8_binary,
                phq8_labels
            ])
    return all_data


if __name__ == '__main__':
    
    num_utt = 5  # 병합할 발화 수
    train_test_split = False # train / test / valid 로 split할지 여부 

    audio_dir = f'/home/dilab/hrimlee/test/daic_woz_process/DAIC_processed_merged/audio_{num_utt}'
    txt_dir = f'/home/dilab/hrimlee/test/daic_woz_process/DAIC_processed_merged/text_{num_utt}'
    
    data_dir = "/home/dilab/hrimlee/test/daic_woz_process/DAIC/" 
    label_file = os.path.join(data_dir, "train_split_Depression_AVEC2017.csv") 
    val_label_file = os.path.join(data_dir, "dev_split_Depression_AVEC2017.csv") 
    
    # Train 데이터 전처리
    train_list = preprocessing_data(audio_dir, txt_dir, label_file)
    
    # Validation 데이터 전처리
    val_list = preprocessing_data(audio_dir, txt_dir, val_label_file)
    
    # 결과 출력
    # Total num Train Data: 3220 
    print("Total num Train Data:", len(train_list))
    
    # Total num Validation Data: 1254
    print("Total num Validation Data:", len(val_list))
    
    # [Split dataset with only Train & Test set]. 
    if train_test_split: 
        output_dir = os.path.join('plots')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # train_list의 participant_id 개수: 101
        # 318, 321, 341, 362, 409 제외 (Out of sync)
        participant_ids_train = [item[0] for item in train_list]
        unique_participant_ids_train = set(participant_ids_train)
        num_unique_participants_train = len(unique_participant_ids_train)
        print(f"Number of unique participant IDs in train_list: {num_unique_participants_train}")
        
        # val_list의 participant_id 개수: 33
        # 451, 458 제외 (NO ellie)
        participant_ids_val = [item[0] for item in val_list]
        unique_participant_ids_val = set(participant_ids_val)
        num_unique_participants_val = len(unique_participant_ids_val)
        print(f"Number of unique participant IDs in val_list: {num_unique_participants_val}")
    
        # 각 데이터셋에 대해 클래스별 개수 계산
        train_counts = count_classes(train_list)
        test_counts = count_classes(val_list)
        
        print_counts(train_counts, "Train Data")
        print_counts(test_counts, "Test Data")

        
        # 그래프 그리기 및 저장
        # plot_counts(train_counts, "Train Data", output_dir)
        # plot_counts(test_counts, "Test Data", output_dir)
        plot_combined_counts(train_counts, None, test_counts, output_dir)
            
        # Combine train and validation data into a single dictionary
        train_test_split = {"train": train_list, "test": val_list}
        
        
        # Save the combined data into a JSON file
        output_json_path = os.path.join(data_dir, 'train_test_split.json')
        with open(output_json_path, 'w') as json_file:
            json.dump(train_test_split, json_file, indent=4)
        
        print(f"Train/test split data saved to {output_json_path}")
        
    # [Split dataset with  Train & Valid & Test set]. 
    else:
        test_list = val_list 
        
        output_dir = os.path.join('tvt_plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Participant ID와 해당하는 클래스 라벨(phq8_binary) 추출
        participant_id_to_label = {}
        for item in train_list:
            participant_id = item[0]
            phq8_binary = item[3]
            participant_id_to_label[participant_id] = phq8_binary

        # 고유한 Participant ID와 그에 해당하는 라벨 리스트 생성
        participant_ids = list(participant_id_to_label.keys())
        labels = [participant_id_to_label[pid] for pid in participant_ids]

        # StratifiedShuffleSplit을 사용하여 80% Train, 20% Validation 분할
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_indices, val_indices = next(sss.split(participant_ids, labels))

        # Train 및 Validation Participant ID 리스트 생성
        train_ids = [participant_ids[idx] for idx in train_indices]
        val_ids = [participant_ids[idx] for idx in val_indices]

        # 새로운 train_list와 val_list 생성
        train_list_new = [item for item in train_list if item[0] in train_ids]
        val_list_new = [item for item in train_list if item[0] in val_ids]

        # train_list와 val_list 업데이트
        train_list = train_list_new
        val_list = val_list_new
    
        # 결과 출력
        # train_list의 participant_id 개수 세기
        participant_ids_train = [item[0] for item in train_list]
        unique_participant_ids_train = set(participant_ids_train)
        num_unique_participants_train = len(unique_participant_ids_train)
        print(f"Number of unique participant IDs in train_list: {num_unique_participants_train}")
        
        # val_list의 participant_id 개수 세기
        participant_ids_val = [item[0] for item in val_list]
        unique_participant_ids_val = set(participant_ids_val)
        num_unique_participants_val = len(unique_participant_ids_val)
        print(f"Number of unique participant IDs in val_list: {num_unique_participants_val}")
        
        # test_list의 participant_id 개수 세기
        participant_ids_test = [item[0] for item in test_list]
        unique_participant_ids_test = set(participant_ids_test)
        num_unique_participants_test = len(unique_participant_ids_test)
        print(f"Number of unique participant IDs in test_list: {num_unique_participants_test}")
        
        train_counts = count_classes(train_list)
        val_counts = count_classes(val_list)
        test_counts = count_classes(test_list)
        
        print_counts(train_counts, "Train Data")
        print_counts(val_counts, "Validation Data")
        print_counts(test_counts, "Test Data")
        
        # 그래프 그리기 및 저장
        # plot_counts(train_counts, "Train Data", output_dir)
        # plot_counts(val_counts, "Validation Data", output_dir)
        # plot_counts(test_counts, "Test Data", output_dir)
        plot_combined_counts(train_counts, val_counts, test_counts, output_dir)
                
        # Combine train, val, and test data into a single dictionary
        train_test_split = {"train": train_list, "dev": val_list, "test": test_list}
        
        # Save the combined data into a JSON file
        output_json_path = os.path.join(data_dir, 'train_val_test_split.json')
        with open(output_json_path, 'w') as json_file:
            json.dump(train_test_split, json_file, indent=4)
        
        print(f"Train/val/test split data saved to {output_json_path}")
