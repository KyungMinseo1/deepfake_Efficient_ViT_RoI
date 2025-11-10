import argparse, yaml, os, torch, collections, cv2, sys, json
from pathlib import Path
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from cross_efficient_vit import CrossEfficientViT
import pandas as pd
from utils2 import shuffle_dataset, get_n_params
from tqdm import tqdm
import numpy as np
class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

with SuppressOutput():
    from mediapipe.python.solutions import face_mesh
    
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DF40_DIR = os.path.join(DATA_DIR, "DF40")
MODELS_PATH = "models"
METADATA_PATH = os.path.join(DATA_DIR, "dataset_json")
SAMPLE_DATA_PATH = r'data\DF40\blendface\frames\001_870\000.png'

def extract_paths(folder_lst, train_paths, train_label, val_paths, val_label):
    """
    train_paths: list # [[path, path, ...], [path, path, ...], ]
    train_label: list # [0, 1, 0, 1, ...]
    test_paths: list
    test_label: list
    """
    for f in tqdm(folder_lst):
        json_path = f + '_ff.json'
        meta_keys = f + '_ff'
        with open(os.path.join(METADATA_PATH, json_path), 'r') as ff:
            metadata = json.load(ff)
        exact_dir_path = {
            ed.split('_')[0] : ed
            for ed in os.listdir(os.path.join(DF40_DIR, f, 'frames'))
        }
        for r_f in tqdm(metadata[meta_keys].keys()):                          # ~~~_Real vs ~~~_Fake
            for t_t in metadata[meta_keys][r_f].keys():                 # train vs test
                for f_name in metadata[meta_keys][r_f][t_t].keys():     # each file number (953)
                    subpath_lst = []
                    no_files = False
                    for path in metadata[meta_keys][r_f][t_t][f_name]['frames']:
                        path = Path(path)
                        parts = path.parts
                        idx = parts.index('frames')
                        file_id = os.path.join(*parts[idx+1:idx+2])
                        if file_id in exact_dir_path.keys():
                            exact_dir = exact_dir_path[file_id]
                        else:
                            no_files = True
                            break
                        frame_name = os.path.join(*parts[idx+2:idx+3])
                        final_path = os.path.join(DF40_DIR, f, 'frames', exact_dir, frame_name) # data/DF40/blendface/frames/001/071/277.png # type: ignore
                        subpath_lst.append(final_path)
                    if t_t == 'train' and not no_files:
                        train_paths.append(subpath_lst)
                        if 'Real' in r_f:
                            train_label.append(0)
                        else:
                            train_label.append(1)
                    elif t_t == 'test' and not no_files:
                        val_paths.append(subpath_lst)
                        if 'Real' in r_f:
                            val_label.append(0)
                        else:
                            val_label.append(0)
                    else:
                        continue


if __name__=="__main__":
    folder_lst = ['blendface', 'danet']
    train_paths, train_label, val_paths, val_label = [], [], [], []
    extract_paths(folder_lst, train_paths=train_paths, train_label=train_label, val_paths=val_paths, val_label=val_label)
    print(len(train_paths), len(train_label), len(val_paths), len(val_label))