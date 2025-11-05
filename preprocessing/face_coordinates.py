import argparse
import json
import os
import sys
from os import cpu_count

sys.stderr = open(os.devnull, 'w')

class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GLOG_minloglevel"] = "3"   # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL (2->3으로 변경)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # (2->3으로 변경)

from functools import partial
from multiprocessing.pool import Pool

import cv2
with SuppressOutput():
    from mediapipe.python.solutions import face_mesh

from utils import get_crop_paths, get_method_from_name

from absl import logging
logging.set_verbosity(logging.ERROR)

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm

def extract_coordinates(cropped_img, output_dir):
    try:
        data_folder = get_method_from_name(cropped_img)
        inner_folder = os.path.basename(os.path.dirname(cropped_img))
        id = os.path.splitext(os.path.basename(cropped_img))[0]
        if not os.path.exists(cropped_img):
            return cropped_img
        
        result = []

        image = cv2.imread(cropped_img)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # FaceMesh 초기화 시에도 stderr 억제
        with SuppressOutput():
            mesh = face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5)
            
        with mesh:
            results = mesh.process(rgb_image)

            if results.multi_face_landmarks: # type: ignore
                for face_landmarks in results.multi_face_landmarks: # type: ignore
                    for idx, lm in enumerate(face_landmarks.landmark[::4]):
                        x = int(lm.x * image.shape[1])
                        y = int(lm.y * image.shape[0])
                        result.append([x, y])

        os.makedirs(os.path.join(output_dir, data_folder, inner_folder), exist_ok=True)
        # ex) data/archive/FaceForensics++_C23/coordinates/DeepFakeDetection/01_02_meeting_serious_YYGY8LOK
        output_path = os.path.join(output_dir, data_folder, inner_folder, id+'.json')
        if len(result) > 0:
            with open(output_path, "w") as coordinates_f:
                json.dump(result, coordinates_f)
            return
        else:
            return cropped_img

    except Exception as e:
        print("Error:", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='', type=str,
                        help='Crops directory')
    # data/archive/FaceForensics++_C23/crops
    parser.add_argument('--output_path', default='', type=str,
                        help='Output directory')
    # data/archive/FaceForensics++_C23/coordinates
    opt = parser.parse_args()
    print(opt)

    os.makedirs(opt.output_path, exist_ok=True)

    excluded_crops = []
    for data_folder in os.listdir(opt.output_path):
        data_folder_path = os.path.join(opt.output_path, data_folder)
        inner_folder_lst = os.listdir(data_folder_path)
        for inner_folder in inner_folder_lst:
            inner_folder_path = os.path.join(data_folder_path, inner_folder)
            excluded_crop_lst = os.listdir(inner_folder_path)
            for ex_crop in excluded_crop_lst:
                excluded_crops.append(os.path.join(opt.data_path, data_folder, inner_folder, os.path.splitext(os.path.basename(ex_crop))[0]+'.png'))

    crops_paths = get_crop_paths(opt.data_path, excluded_crops)

    with Pool(processes=cpu_count()-2) as p: # type: ignore
        with tqdm(total=len(crops_paths)) as pbar:
            for v in p.imap_unordered(partial(extract_coordinates, output_dir=opt.output_path), crops_paths):
                pbar.update()
                
if __name__ == "__main__":
    main()