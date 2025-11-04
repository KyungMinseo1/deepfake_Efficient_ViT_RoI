import cv2
from albumentations import Compose, PadIfNeeded, LongestMaxSize
import numpy as np
import os
import cv2
import torch
from statistics import mean

def transform_frame(image, image_size):
    transform_pipeline = Compose([
                LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
                PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_REPLICATE)
                ]
            )
    return transform_pipeline(image=image)['image']
    
    
def resize(image, image_size):
    try:
        return cv2.resize(image, dsize=(image_size, image_size))
    except:
        return []

def custom_round(values):
    result = []
    for value in values:
        if value > 0.55:
            result.append(1)
        else:
            result.append(0)
    return np.asarray(result)

def custom_video_round(preds):
    for pred_value in preds:
        if pred_value > 0.55:
            return pred_value
    return mean(preds)

        

def get_method(video, data_path):
    methods = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
    selected_method = ""
    for method in methods:
        if method in video:
            selected_method = method
            break
    return selected_method

def shuffle_dataset(dataset):
  import random
  random.seed(4)
  random.shuffle(dataset)
  return dataset

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
def check_correct(preds, labels):
    preds = preds.cpu()
    labels = labels.cpu()
    preds = [np.asarray(torch.sigmoid(pred).detach().numpy()).round() for pred in preds]

    correct = 0
    positive_class = 0
    negative_class = 0
    for i in range(len(labels)):
        pred = int(preds[i])
        if labels[i] == pred:
            correct += 1
        if pred == 1:
            positive_class += 1
        else:
            negative_class += 1
    return correct, positive_class, negative_class
