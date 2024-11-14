import json
import yaml
import numpy as np
import pandas as pd
import pickle, pdb, re

from tqdm import tqdm
from pathlib import Path
import os
'''
NOTE:
기존 iemocap, iemocap6 txt파일 경로 수정.
csv 파일에서 한줄씩 읽어올수있도록 전처리 바꿔놔서,
json 파일 텍스트 경로 수정 
'''



if __name__ == '__main__':
    
    import json

    # JSON 파일 로드
    with open('/home/dilab/hrimlee/peft-ser/train_split/iemocap6_fold1.json', 'r') as file: 
        data = json.load(file)

    # dict 항목들 (train, test, dev 등)을 순회하며 네 번째 항목 수정
    for dict_name, items in data.items():
        for item in items:
            item[4] = f"/home/dilab/hrimlee/peft-ser/data/IEMOCAP_{dict_name}.csv"

    # 수정된 데이터를 다시 JSON 파일로 저장
    os.makedirs('../train_split/train_split_session', exist_ok=True)
    with open('../train_split/train_split_session/iemocap6_fold1.json', 'w') as file: 
        json.dump(data, file, indent=4)

    print("파일이 성공적으로 수정되었습니다.")
        
    # # dump the dictionary
    # jsonString = json.dumps(return_dict, indent=4)
    # jsonFile = open(str(output_path.joinpath('train_split', f'iemocap_fold{1}.json')), "w")
    # jsonFile.write(jsonString)
    # jsonFile.close()


