from tqdm import tqdm
import os, json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DF40_DIR = os.path.join(DATA_DIR, "DF40")
TEST_DF40_DIR = os.path.join(DATA_DIR, "DF40_test")
MODELS_PATH = "models"
METADATA_PATH = os.path.join(DATA_DIR, "dataset_json")
SAMPLE_DATA_PATH = r'data\DF40\blendface\frames\001_870\000.png'

def extract_paths_for_DF40(folder_lst, real_path, fake_path, real_data, fake_data):
    """
    train_paths: list # [[path, path, ...], [path, path, ...], ]
    train_label: list # [0, 1, 0, 1, ...]
    test_paths: list
    test_label: list
    """
    for f in tqdm(folder_lst):
        json_path = f + '_ff.json'
        if f == "RDDM": # except for RDDM
            meta_keys = "rddm_ff"
        else:
            meta_keys = f + '_ff'
        with open(os.path.join(METADATA_PATH, json_path), 'r') as ff:
            metadata = json.load(ff)
        exact_dir_path = {
            (ed.split('_')[0] if '_' in ed else ed) : ed
            for ed in os.listdir(os.path.join(DF40_DIR, f, 'frames'))
        }
        for r_f in tqdm(metadata[meta_keys].keys()):                # ~~~_Real vs ~~~_Fake
            for t_t in metadata[meta_keys][r_f].keys():             # train vs test
                for f_name in metadata[meta_keys][r_f][t_t].keys(): # each file number (953)
                    try:
                        subpath = None
                        for path in metadata[meta_keys][r_f][t_t][f_name]['frames']:
                            path = Path(path)
                            parts = path.parts
                            if 'frames' in parts:
                                idx = parts.index('frames')
                            elif 'ff' in parts:
                                idx = parts.index('ff')
                            file_id = os.path.join(*parts[idx+1:idx+2])
                            if '_' in file_id:
                                file_id = file_id.split('_')[0]
                            if file_id in exact_dir_path.keys():
                                exact_dir = exact_dir_path[file_id]
                            else:
                                print(f"{f}-{file_id} Wrong key")
                                break
                            if t_t == "train":
                                final_path = os.path.join(DF40_DIR, f, 'frames', exact_dir) # data/DF40/blendface/frames/071
                            else:
                                final_path = os.path.join(TEST_DF40_DIR, f, 'ff', 'frames', exact_dir)
                            if not os.path.isdir(final_path):
                                print(f"{final_path} 폴더 없음")
                            subpath = final_path
                            # 한 개만 진행
                            break
                        if 'Real' in r_f and subpath is not None:
                            real_path.append(subpath)
                            real_data.append(f)
                        elif 'Fake' in r_f and subpath is not None:
                            fake_path.append(subpath)
                            fake_data.append(f)
                        else:
                            continue
                    except Exception as e:
                        print(f"{f}-{r_f}-{t_t}-{f_name} 오류", e)
                        continue

def extract_paths_for_DF40_test(folder_lst, real_path, fake_path, real_data, fake_data):
    """
    train_paths: list # [[path, path, ...], [path, path, ...], ]
    train_label: list # [0, 1, 0, 1, ...]
    test_paths: list
    test_label: list
    """
    for f in tqdm(folder_lst):
        json_path = f + '_ff.json'
        if f == "RDDM": # except for RDDM
            meta_keys = "rddm_ff"
        else:
            meta_keys = f + '_ff'
        with open(os.path.join(METADATA_PATH, json_path), 'r') as ff:
            metadata = json.load(ff)
        exact_dir_path = {
            (ed.split('_')[0] if '_' in ed else ed) : ed
            for ed in os.listdir(os.path.join(TEST_DF40_DIR, f, 'ff', 'frames'))
        }
        for r_f in tqdm(metadata[meta_keys].keys()):                # ~~~_Real vs ~~~_Fake
            for t_t in metadata[meta_keys][r_f].keys():             # train vs test
                for f_name in metadata[meta_keys][r_f][t_t].keys(): # each file number (953)
                    try:
                        subpath = None
                        for path in metadata[meta_keys][r_f][t_t][f_name]['frames']:
                            path = Path(path)
                            parts = path.parts
                            if 'frames' in parts:
                                idx = parts.index('frames')
                            elif 'ff' in parts:
                                idx = parts.index('ff')
                            file_id = os.path.join(*parts[idx+1:idx+2])
                            if '_' in file_id:
                                file_id = file_id.split('_')[0]
                            if file_id in exact_dir_path.keys():
                                exact_dir = exact_dir_path[file_id]
                            else:
                                print(f"{f}-{file_id} Wrong key")
                                break
                            if t_t == "train":
                                final_path = os.path.join(DF40_DIR, f, 'frames', exact_dir) # data/DF40/blendface/frames/071
                            else:
                                final_path = os.path.join(TEST_DF40_DIR, f, 'ff', 'frames', exact_dir)
                            if not os.path.isdir(final_path):
                                print(f"{final_path} 폴더 없음")
                            subpath = final_path
                            # 한 개만 진행
                            break
                        if 'Real' in r_f and subpath is not None:
                            real_path.append(subpath)
                            real_data.append(f)
                        elif 'Fake' in r_f and subpath is not None:
                            fake_path.append(subpath)
                            fake_data.append(f)
                        else:
                            continue
                    except Exception as e:
                        print(f"{f}-{r_f}-{t_t}-{f_name} 오류", e)
                        continue


if __name__=="__main__":
    folder_lst = os.listdir(DF40_DIR)
    real_path = []
    fake_path = []
    real_data = []
    fake_data = []
    extract_paths_for_DF40(folder_lst, real_path, fake_path, real_data, fake_data)
    extract_paths_for_DF40_test(folder_lst, real_path, fake_path, real_data, fake_data)
    print(len(real_path), len(fake_path))
    print(real_path[:5], fake_path[:5])
    print(np.unique(real_data, return_counts=True), np.unique(fake_data, return_counts=True))
    real_df = pd.DataFrame({'Path':real_path,'Data':real_data})
    fake_df = pd.DataFrame({'Path':fake_path,'Data':fake_data})
    print(real_df.head())
    print(fake_df.head())
    real_df.to_csv(os.path.join(DATA_DIR, 'csv', 'real_path.csv'), index=False)
    fake_df.to_csv(os.path.join(DATA_DIR, 'csv', 'fake_path.csv'), index=False)
    print('CSV saved')
