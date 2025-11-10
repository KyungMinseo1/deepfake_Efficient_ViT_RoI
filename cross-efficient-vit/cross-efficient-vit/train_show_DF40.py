import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
from cross_efficient_vit import CrossEfficientViT
import cv2
from tqdm import tqdm
from utils2 import check_correct, shuffle_dataset, get_n_params
from torch.optim import lr_scheduler
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
import random
from pathlib import Path
from PIL import Image

import os
import sys
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

from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, \
    LongestMaxSize, KeypointParams

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["GLOG_minloglevel"] = "3"   # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL (2->3으로 변경)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # (2->3으로 변경)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DF40_DIR = os.path.join(DATA_DIR, "DF40")
TEST_DF40_DIR = os.path.join(DATA_DIR, "DF40_test")
MODELS_PATH = "models"
METADATA_PATH = os.path.join(DATA_DIR, "dataset_json")
SAMPLE_DATA_PATH = r'data\DF40\blendface\frames\001_870\000.png'

left_eye = [33, 133, 144, 153, 158, 160]
right_eye = [263, 362, 373, 380, 385, 387]
nose = [(168, 6, 197, 195, 5, 4, 1, 94), (20, 250), (360, 131), (48, 278), (166, 392), (99, 326)]
lips = [(0, 17), (39, 269), (185, 409), (181, 405),13, (146, 375), (81, 311), (191, 415)]
left_cheek = [123, 205, 192, 187, 206, 214, 216, 143, 117, 119]
right_cheek = [352, 425, 416, 411, 426, 434, 436, 372, 346, 348]
left_eye_brow = [46, 52, 53, 65]
right_eye_brow = [295, 283, 276, 282]
jaw = [214, 204, 201, 421, 424, 434]
outer = [137, 215, 135, 170, 171, 396, 395, 364, 435, 366, 264, 301, 333, 337, 108, 104, 71, 34]

target_parts = [left_eye, right_eye, nose, lips, left_cheek, right_cheek, left_eye_brow, right_eye_brow, jaw, outer]

index = []

for t in target_parts:
    for i in t:
        if isinstance(i, int):
            index.append(i)
        else:
            for ii in i:
                index.append(ii)

def match_size(size):
    keypoint_params = KeypointParams(format='xy', remove_invisible=False) 
    return Compose([
        LongestMaxSize(max_size=size, interpolation=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT)
        ], keypoint_params=keypoint_params)

def extract_paths(folder_lst, train_paths, train_label, val_paths, val_label):
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
                        subpath_lst = []
                        no_files = False
                        for path in metadata[meta_keys][r_f][t_t][f_name]['frames']:
                            path = Path(path)
                            parts = path.parts
                            if 'frames' in parts:
                                idx = parts.index('frames')
                            elif 'ff' in parts:
                                idx = parts.index('ff')
                            file_id = os.path.join(*parts[idx+1:idx+2])
                            if file_id in exact_dir_path.keys():
                                exact_dir = exact_dir_path[file_id]
                            else:
                                no_files = True
                                break
                            frame_name = os.path.join(*parts[idx+2:idx+3])
                            if t_t == "train":
                                final_path = os.path.join(DF40_DIR, f, 'frames', exact_dir, frame_name) # data/DF40/blendface/frames/071/277.png
                            else:
                                final_path = os.path.join(TEST_DF40_DIR, f, 'ff', 'frames', exact_dir, frame_name)
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
                    except Exception as e:
                        print(f"{f}-{r_f}-{t_t}-{f_name} 오류", e)
                        continue

def read_frames(data, dataset, config, mode='train', is_sample=False):
    '''
    data -> (crops_path, label)
    crops_path -> [exact_path, exact_path, ...]
    label -> 0 or 1
    train_dataset -> list()
    validation_dataset -> list()
    mode -> 'train' or 'validation'
    '''

    crops_path, label = data

    if not is_sample:
        min_video_frames = max(int(config['training']['frames-per-video']),1)
        crops_path = random.sample(crops_path, min_video_frames)

    if is_sample:
        print('crops_path: ', crops_path)
        image = cv2.imread(os.path.join(crops_path)) # type: ignore
        if image is None:
            print("⚠️ 이미지 파일을 찾을 수 없습니다. 경로를 다시 확인하세요.")
            return
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
        coordinates = []
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
                    for lm in np.array(face_landmarks.landmark[:])[index]:
                        x = int(lm.x * image.shape[1])
                        y = int(lm.y * image.shape[0])
                        coordinates.append([x, y])
        if not len(coordinates) == 97:
            return
        transform = match_size(config['model']['image-size'])
        augmented = transform(image=image, keypoints=coordinates)
        image = augmented['image']
        coordinates = augmented['keypoints']
        if image is not None:
            dataset.append((image, label, coordinates))
        else:
            print("⚠️ 이미지 파일을 찾을 수 없습니다. 경로를 다시 확인하세요.")
    else:
        """
        # Calculate the interval to extract the frames
        frames_number = len(os.listdir(crops_path))
        if label == 0:
            min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-real']),1) # Compensate unbalancing
        else:
            min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing-fake']),1)

        if mode == 'validation':
            min_video_frames = int(max(min_video_frames/8, 2))
        frames_interval = int(frames_number / min_video_frames)
        frames_paths = [f for f in os.listdir(crops_path) if f.endswith('.png')]
        frames_paths_dict = {}

        # Group the faces with the same index, reduce probabiity to skip some faces in the same video
        for path in frames_paths:
            # 각각의 프레임별 얼굴사진에 대하여
            for i in range(0,1):
                if "_" + str(i) in path:
                    if i not in frames_paths_dict.keys():
                        frames_paths_dict[i] = [path]
                    else:
                        frames_paths_dict[i].append(path)
                        # frames_paths_dict ex) {0 : [프레임 사진 이름, 프레임 사진 이름, ...]}

        # Select only the frames at a certain interval
        if frames_interval > 0:
            for key in frames_paths_dict.keys():
                if len(frames_paths_dict[key]) > frames_interval:
                    frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
                
                frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]

        # Select N frames from the collected ones
        for key in frames_paths_dict.keys():
            for frame_image in frames_paths_dict[key]:
                #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
                image = cv2.imread(os.path.join(crops_path, frame_image))
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
                coordinates = []
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
                            for lm in np.array(face_landmarks.landmark[:])[index]:
                                x = int(lm.x * image.shape[1])
                                y = int(lm.y * image.shape[0])
                                coordinates.append([x, y])
                if not len(coordinates) == 97:
                    return
                transform = match_size(config['model']['image-size'])
                augmented = transform(image=image, keypoints=coordinates)
                image = augmented['image']
                coordinates = augmented['keypoints']
                if image is not None:
                    dataset.append((image, label, coordinates))
                else:
                    print("⚠️ 이미지 파일을 찾을 수 없습니다. 경로를 다시 확인하세요.")
        """
        for frame_image in crops_path:
            if not os.path.exists(frame_image):
                print(f"⚠️ 파일이 존재하지 않음: {frame_image}")
                return
            try:
                pil_image = Image.open(frame_image)
            except Exception as e:
                print(f"🚨 이미지 열기 오류: {frame_image}, 오류: {e}")
                return
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # type: ignore
            if image is None:
                print(f"⚠️ {frame_image}의 이미지 파일을 읽을 수 없습니다.")
                return
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            coordinates = []
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
                        for lm in np.array(face_landmarks.landmark[:])[index]:
                            x = int(lm.x * image.shape[1])
                            y = int(lm.y * image.shape[0])
                            coordinates.append([x, y])
            if not len(coordinates) == 97:
                return
            transform = match_size(config['model']['image-size'])
            augmented = transform(image=image, keypoints=coordinates)
            image = augmented['image']
            coordinates = augmented['keypoints']
            if image is not None:
                dataset.append((image, label, coordinates))
            else:
                print(f"⚠️ {frame_image}의 이미지 파일을 찾을 수 없습니다. 경로를 다시 확인하세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='All', 
                        help="""Which dataset to use (blendface|danet|ddim|DiT|e4s|facedancer|faceswa|facevid2vid|fomm|fsgan|hyperreenact|inswap
                                |lia|mcnet|mobileswap|MRAA|one_shot_free|pirender|pixart|rddm|sadtalker|sd2.1|simswap|SiT|StyleGAN2|StyleGAN3
                                |StyleGANXL|tpsm|uniface|VQGAN|wav2lip) or Sample (seperate each dataset with comma)""")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, default='cross-efficient-vit/cross-efficient-vit/configs/architecture.yaml',
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5, 
                        help="How many epochs wait before stopping for validation loss not improving.")
    
    opt = parser.parse_args()
    print(opt)

    mgr = Manager()
    train_dataset = mgr.list()
    validation_dataset = mgr.list()
    train_paths, train_label, val_paths, val_label = [], [], [], []

    with open(opt.config, 'r', encoding="utf-8") as ymlfile:
        config = yaml.safe_load(ymlfile)
 
    if opt.dataset != "Sample":
        model = CrossEfficientViT(config=config, efficient_net=opt.efficient_net)
        model.train()
        is_sample = False
        print("MODEL ON")
    else:
        is_sample = True
        model = CrossEfficientViT(config=config, is_sample=is_sample, efficient_net=opt.efficient_net)
        model.eval()
        print("SAMPLE MODEL ON")

    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 학습 히스토리 초기화
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'train_positive': [],
        'train_negative': [],
        'val_positive': [],
        'val_negative': []
    }
    
    starting_epoch = 0
    if opt.resume and os.path.exists(opt.resume):
        print(f"Loading checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        
        print(f"Resuming from epoch {starting_epoch}")
        print(f"Previous best val_loss: {min(history['val_loss']) if history['val_loss'] else 'N/A'}")
    else:
        print("No checkpoint loaded. Starting from scratch.")

    print("Model Parameters:", get_n_params(model))

    #READ DATASET
    if opt.dataset == "Sample":
        train_data = list(zip([SAMPLE_DATA_PATH], [0]))
        val_data = None
        print("You are currently looking at Sample DATA")
    elif opt.dataset != "All":
        folders = opt.dataset.split(',')
        print('Target Folders: ', folders)
        extract_paths(folders, train_paths=train_paths, train_label=train_label, val_paths=val_paths, val_label=val_label)
        train_data = list(zip(train_paths, train_label))
        val_data = list(zip(val_paths, val_label))

    else:
        folders = os.listdir(DF40_DIR)
        print('Target Folders: ', folders)
        extract_paths(folders, train_paths=train_paths, train_label=train_label, val_paths=val_paths, val_label=val_label)
        train_data = list(zip(train_paths, train_label))
        val_data = list(zip(val_paths, val_label))
        
    with Pool(processes=10) as p:
        with tqdm(total=len(train_data)) as pbar:
            for v in p.imap_unordered(partial(read_frames, dataset=train_dataset, config=config, is_sample = is_sample),train_data):
                pbar.update()
        if val_data:
            with tqdm(total=len(val_data)) as pbar:
                for v in p.imap_unordered(partial(read_frames, dataset=validation_dataset, config=config, mode='validation', is_sample = is_sample),val_data):
                    pbar.update()
    train_samples = len(train_dataset)
    train_dataset = shuffle_dataset(train_dataset)
    validation_samples = len(validation_dataset)
    if validation_samples > 0:
        validation_dataset = shuffle_dataset(validation_dataset)

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(train_counters)
    
    if 0 in train_counters and 1 in train_counters and train_counters[1] != 0:
        class_weights = train_counters[0] / train_counters[1]
    else:
        class_weights = 1.0  # 기본값
        print("⚠️ Warning: Some classes missing — setting class_weights = 1.0")
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    if validation_samples > 0:
        val_counters = collections.Counter(image[1] for image in validation_dataset)
        print(val_counters)
    else:
        print("Skipping Validation Count")
    print("___________________")

    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]).to(device))

    # Create the data loaders
    if validation_samples > 0:
        validation_labels = np.asarray([row[1] for row in validation_dataset])
    labels = np.asarray([row[1] for row in train_dataset])

    train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset]), np.asarray([row[2] for row in train_dataset]), labels, config['model']['image-size'])
    dl = DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=True, drop_last=False, timeout=0,
                                 worker_init_fn=None, persistent_workers=False)
    del train_dataset

    if validation_samples > 0:
        validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset]), np.asarray([row[2] for row in validation_dataset]), validation_labels, config['model']['image-size'], mode='validation')
        val_dl = DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=False, sampler=None,
                                        batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                        pin_memory=True, drop_last=False, timeout=0,
                                        worker_init_fn=None, persistent_workers=False)
    del validation_dataset

    model = model.to(device)
    print("Current Device:", device)
    
    if is_sample:
        for index, (images, coordinates, labels) in enumerate(dl):
            labels = labels.unsqueeze(1).to(device)
            images = images.to(device)
            coordinates = coordinates.to(device)
            
            y_pred = model(images, coordinates)

        print("Visualization Finished!")

    else:
        counter = 0
        not_improved_loss = 0
        previous_loss = math.inf if not history['val_loss'] else history['val_loss'][-1]

        for t in range(starting_epoch, opt.num_epochs + 1):
            if not_improved_loss == opt.patience:
                break
            counter = 0

            # ===== Training Phase =====
            model.train()
            total_loss = 0
            
            bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*config['training']['bs'])+len(val_dl))
            train_correct = 0
            positive = 0
            negative = 0
            
            for index, (images, coordinates, labels) in enumerate(dl):
                labels = labels.unsqueeze(1).to(device)
                images = images.to(device)
                coordinates = coordinates.to(device)
                            
                optimizer.zero_grad()
                y_pred = model(images, coordinates)
                loss = loss_fn(y_pred, labels)
                loss.backward()
                optimizer.step()
            
                corrects, positive_class, negative_class = check_correct(
                    y_pred.detach().cpu(), 
                    labels.detach().cpu()
                )  
                train_correct += corrects
                positive += positive_class
                negative += negative_class
                
                counter += 1
                total_loss += loss.item()
                
                for i in range(config['training']['bs']):
                    bar.next()
                
                if index % 1200 == 0:
                    print(f"\nLoss: {total_loss/counter:.4f}, Accuracy: {train_correct/(counter*config['training']['bs']):.4f}, Train 0s: {negative}, Train 1s: {positive}")

            train_correct /= train_samples
            total_loss /= counter

            # ===== Validation Phase =====
            model.eval()
            total_val_loss = 0
            val_counter = 0
            val_correct = 0
            val_positive = 0
            val_negative = 0

            with torch.no_grad():
                for index, (val_images, val_coordinates, val_labels) in enumerate(val_dl):
                    val_labels = val_labels.unsqueeze(1).to(device)
                    val_images = val_images.to(device)
                    val_coordinates = val_coordinates.to(device)
                    
                    val_pred = model(val_images, val_coordinates)
                    val_loss = loss_fn(val_pred, val_labels)
                    
                    total_val_loss += val_loss.item()
                    
                    # CPU에서 metric 계산
                    corrects, positive_class, negative_class = check_correct(
                        val_pred.cpu(), 
                        val_labels.cpu()
                    )
                    val_correct += corrects
                    val_positive += positive_class
                    val_negative += negative_class
                    val_counter += 1
                    bar.next()
                
            scheduler.step()
            bar.finish()
            
            total_val_loss /= val_counter
            val_correct /= validation_samples
            
            if previous_loss <= total_val_loss:
                print("Validation loss did not improve")
                not_improved_loss += 1
            else:
                not_improved_loss = 0
            
            previous_loss = total_val_loss
            
            # 히스토리에 기록
            history['train_loss'].append(float(total_loss))
            history['train_accuracy'].append(float(train_correct))
            history['val_loss'].append(float(total_val_loss))
            history['val_accuracy'].append(float(val_correct))
            history['train_positive'].append(int(positive))
            history['train_negative'].append(int(negative))
            history['val_positive'].append(int(val_positive))
            history['val_negative'].append(int(val_negative))
            
            print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
                str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(np.count_nonzero(validation_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(np.count_nonzero(validation_labels == 1)))

            
            if not os.path.exists(MODELS_PATH):
                os.makedirs(MODELS_PATH)
            
            # 완전한 체크포인트 저장
            checkpoint = {
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'config': config,
                'dataset': opt.dataset
            }
            
            checkpoint_path = os.path.join(MODELS_PATH, f"checkpoint_epoch{t}_{opt.dataset}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # 최고 성능 모델 별도 저장
            if len(history['val_loss']) > 0 and total_val_loss == min(history['val_loss']):
                best_path = os.path.join(MODELS_PATH, f"best_model_{opt.dataset}.pth")
                torch.save(checkpoint, best_path)
                print(f"Best model saved: {best_path}")

        # 학습 완료 후 그래프 생성
        print("\nGenerating training history plots...")

        # 디렉토리 생성
        plots_dir = os.path.join(MODELS_PATH, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        epochs_range = range(starting_epoch, starting_epoch + len(history['train_loss']))

        # Loss 그래프
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
        plt.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy 그래프
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_accuracy'], 'b-', label='Train Accuracy')
        plt.plot(epochs_range, history['val_accuracy'], 'r-', label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'training_history_{opt.dataset}.png'), dpi=300)
        print(f"Plot saved: {os.path.join(plots_dir, f'training_history_{opt.dataset}.png')}")

        # Class distribution 그래프
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_positive'], 'g-', label='Train Positives')
        plt.plot(epochs_range, history['train_negative'], 'b-', label='Train Negatives')
        plt.xlabel('Epoch')
        plt.ylabel('Count')
        plt.title('Training Class Distribution')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['val_positive'], 'g-', label='Val Positives')
        plt.plot(epochs_range, history['val_negative'], 'b-', label='Val Negatives')
        plt.xlabel('Epoch')
        plt.ylabel('Count')
        plt.title('Validation Class Distribution')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'class_distribution_{opt.dataset}.png'), dpi=300)
        print(f"Plot saved: {os.path.join(plots_dir, f'class_distribution_{opt.dataset}.png')}")

        # 히스토리를 JSON으로 저장
        history_path = os.path.join(MODELS_PATH, f'training_history_{opt.dataset}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"History saved: {history_path}")

        print("\nTraining completed!")
        print(f"Best validation loss: {min(history['val_loss']):.4f}")
        print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")


# 처음: python cross-efficient-vit/cross-efficient-vit/train_show_DF40.py --dataset All --efficient_net 7 --config cross-efficient-vit/cross-efficient-vit/configs/architecture_for_b7.yaml
# 샘플: python cross-efficient-vit/cross-efficient-vit/train_show_DF40.py --dataset Sample --efficient_net 7 --config cross-efficient-vit/cross-efficient-vit/configs/architecture_for_b7.yaml
# 재개: python cross-efficient-vit/cross-efficient-vit/train_show_DF40.py --dataset All --resume models/efficientnet_checkpoint10_All.pth --efficient_net 7 --config cross-efficient-vit/cross-efficient-vit/configs/architecture_for_b7.yaml