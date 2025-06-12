import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from string import ascii_lowercase

# Utility function

def read_csv(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

class SignLangDataset(Dataset):
    """Sign language dataset"""

    def __init__(self, csv_file, root_dir, class_index_map=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = read_csv(os.path.join(root_dir,csv_file))
        self.root_dir = root_dir
        self.class_index_map = class_index_map
        self.transform = transform
        # List of class names in order
        self.class_names = list(map(str, list(range(10)))) + list(ascii_lowercase)

    def __len__(self):
        """
        Calculates the length of the dataset-
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns one sample (dict consisting of an image and its label)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read the image and labels
        image_path = os.path.join(self.root_dir, self.data[idx][1])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Shape of the image should be H,W,C where C=1
        """image = np.expand_dims(image, 0)"""
        # The label is the index of the class name in the list ['0','1',...,'9','a','b',...'z']
        # because we should have integer labels in the range 0-35 (for 36 classes)
        label = self.class_names.index(self.data[idx][0])
        # for the vgg try
        image = np.stack([image] * 3, axis=-1)                        # → (128, 128, 3), dtype=uint8
                
        sample = {'image': image, 'label': label}

        # Apply the specified transform (e.g., VGG preprocessing: resize, convert to tensor, normalize)
        if self.transform:
            sample = self.transform(sample)

        return sample

### I have read in the internet, that one nice method to use the RF model for image classification, 
### is to use a pretrained Convolutional NN like VGG (Visual Geometry Group from OXford university) to extract features.
### These networks are trained on large datasets like ImageNet and have learned to extract meaningful features from images.