import argparse
import json
import os
import numpy as np
from typing import Type

from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import face_detector
from face_detector import VideoDataset
from face_detector import VideoFaceDetector
from utils import get_video_paths, get_method
import argparse
from PIL import Image

def collate_fn_identity(x):
    return x

def process_videos(videos, detector_cls: Type[VideoFaceDetector], opt):
    detector = face_detector.__dict__[detector_cls](device="cuda:0")

    dataset = VideoDataset(videos)
    valid_indices = [i for i, v in enumerate(videos) if not os.path.exists(os.path.join(opt.data_path, "boxes", get_method(v, opt.data_path), f"{os.path.splitext(os.path.basename(v))[0]}.json"))]
    subset_dataset = Subset(dataset, valid_indices)
    loader = DataLoader(subset_dataset, shuffle=False, num_workers=1, batch_size=1, collate_fn=collate_fn_identity)
    missed_videos = []
    for item in tqdm(loader):
        result = {}
        video, indices, frames = item[0]
        method = get_method(video, opt.data_path)
        out_dir = os.path.join(opt.data_path, "boxes", method)

        id = os.path.splitext(os.path.basename(video))[0]

        if os.path.exists(out_dir) and "{}.json".format(id) in os.listdir(out_dir):
            continue
        batches = [frames[i:i + detector._batch_size] for i in range(0, len(frames), detector._batch_size)]
      
        for j, frames_batch in enumerate(batches):
            try:
                pil_frames = [Image.fromarray(f) for f in frames_batch]
                batch_result = detector._detect_faces(pil_frames)
                result.update({int(j * detector._batch_size) + i : b for i, b in zip(indices, batch_result)})
            except Exception as e:
                print(f"[WARNING] Error processing batch {j} of video {id}: {e}")
                missed_videos.append(id)
                continue
        
        os.makedirs(out_dir, exist_ok=True)
        print(len(result))
        if len(result) > 0:
            with open(os.path.join(out_dir, "{}.json".format(id)), "w") as f:
                json.dump(result, f)
        else:
            missed_videos.append(id)

    if len(missed_videos) > 0:
        print("The detector did not find faces inside the following videos:")
        print(id)
        print("We suggest to re-run the code decreasing the detector threshold.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='', type=str,
                        help='Videos directory')
    parser.add_argument("--detector-type", help="Type of the detector", default="FacenetDetector",
                        choices=["FacenetDetector"])
    parser.add_argument("--processes", help="Number of processes", default=1)
    opt = parser.parse_args()
    print(opt)

    videos_paths = []

    videos_paths = get_video_paths(opt.data_path)
    process_videos(videos_paths, opt.detector_type, opt)


if __name__ == "__main__":
    main()
