import argparse
import json
import os
from os import cpu_count
from pathlib import Path

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm

from utils import get_video_paths, get_method, get_method_from_name

def extract_video(video, root_dir, output_dir):
    try:
        data_folder = get_method_from_name(video)
        bboxes_path = os.path.join(root_dir, "boxes", data_folder, os.path.splitext(os.path.basename(video))[0] + ".json")
        if not os.path.exists(bboxes_path) or not os.path.exists(video):
            return
        with open(bboxes_path, "r") as bbox_f:
            bboxes_dict = json.load(bbox_f)
        
        capture = cv2.VideoCapture(video)
        frames_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        counter = 0
        for i in range(frames_num):
            capture.grab()
            #if i % 2 != 0:
            #    continue
            success, frame = capture.retrieve()
            if not success or str(i) not in bboxes_dict:
                continue
            id = os.path.splitext(os.path.basename(video))[0]
            crops = []
            bboxes = bboxes_dict[str(i)]
            if bboxes is None:
                continue
            else:
                counter += 1
            for bbox in bboxes:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
                w = xmax - xmin
                h = ymax - ymin
                p_h = 0
                p_w = 0
                
                #p_h = h // 3
                #p_w = w // 3
                
                #p_h = h // 6
                #p_w = w // 6

                if h > w:
                    p_w = int((h-w)/2)
                elif h < w:
                    p_h = int((w-h)/2)

                crop = frame[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                h, w = crop.shape[:2]
                crops.append(crop)

            os.makedirs(os.path.join(output_dir, data_folder, id), exist_ok=True)
            for j, crop in enumerate(crops):
                cv2.imwrite(os.path.join(output_dir, data_folder, id, "{}_{}.png".format(i, j)), crop)
        if counter == 0:
            print(video, counter)

    except Exception as e:
        print("Error:", e)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='', type=str,
                        help='Videos directory')
    # data/archive/FaceForensics++_C23
    parser.add_argument('--output_path', default='', type=str,
                        help='Output directory')
    # data/archive/FaceForensics++_C23/crops

    opt = parser.parse_args()
    print(opt)
    
    os.makedirs(opt.output_path, exist_ok=True)
    #excluded_videos = os.listdir(os.path.join(opt.output_dir)) # Useful to avoid to extract from already extracted videos
    excluded_videos = os.listdir(opt.output_path)
    paths = get_video_paths(opt.data_path, excluded_videos)

    with Pool(processes=cpu_count()-2) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=opt.data_path, output_dir=opt.output_path), paths):
                pbar.update()
