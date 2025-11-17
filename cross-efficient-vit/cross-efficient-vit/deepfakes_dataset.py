import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2 
import numpy as np

import uuid
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, \
    LongestMaxSize, KeypointParams

class DeepFakesDataset(Dataset):
    def __init__(self, images, coordinates, labels, image_size, mode = 'train'):
        self.x = images
        self.x2 = coordinates
        self.y = torch.from_numpy(labels)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = images.shape[0]
    
    def create_train_transforms(self, size):
        keypoint_params = KeypointParams(format='xy', remove_invisible=False)
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.2), # type: ignore
            GaussNoise(p=0.3),
            # GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            LongestMaxSize(max_size=size, interpolation=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.4),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ], keypoint_params=keypoint_params) # type: ignore
        
    def create_val_transform(self, size):
        keypoint_params = KeypointParams(format='xy', remove_invisible=False) 
        return Compose([
            LongestMaxSize(max_size=size, interpolation=cv2.INTER_CUBIC),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        ], keypoint_params=keypoint_params)

    def __getitem__(self, index):
        image = self.x[index]
        coordinates = self.x2[index]
        label = self.y[index]
        
        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)
                
        #unique = uuid.uuid4()
        #cv2.imwrite("../dataset/augmented_frames/vit_augmentation/square_fda/"+str(unique)+"_"+str(index)+"_original.png", image)
   
        augmented = transform(image=image, keypoints=coordinates)
        transformed_image = augmented['image']
        transformed_coordinates = augmented['keypoints'] # 변환된 키포인트

        transformed_image = np.transpose(transformed_image, (2, 0, 1))

        if len(transformed_coordinates) > 0:
            transformed_coordinates = np.array(transformed_coordinates)
        else:
            transformed_coordinates = np.zeros((0, 2))
        
        #cv2.imwrite("../dataset/augmented_frames/vit_augmentation/square_fda/"+str(unique)+"_"+str(index)+".png", image)
        
        return torch.tensor(transformed_image).float(), torch.tensor(transformed_coordinates).float(), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return self.n_samples

 
