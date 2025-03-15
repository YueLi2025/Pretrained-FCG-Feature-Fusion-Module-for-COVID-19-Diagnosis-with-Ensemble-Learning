"""
COVID-19 Dataset Loader

This module defines a custom PyTorch Dataset class for loading and preprocessing
COVID-19 chest X-ray/CT images for classification tasks.

The dataset supports four classes:
- COVID-19 (covid)
- Lung Opacity (lung_opacity)
- Normal (normal)
- Viral Pneumonia (viral_pneumonia)
"""

import os
from torch.utils.data import Dataset
import random
from PIL import Image


class MyDataset(Dataset):
    """
    Custom Dataset for COVID-19 chest X-ray/CT image classification.
    
    This dataset loads images from a directory structure where each class
    has its own subdirectory. It supports data augmentation through
    transformations and handles both training and validation/testing modes.
    
    Directory structure expected:
    root_dir/
        covid/
            image1.png
            image2.png
            ...
        lung_opacity/
            image1.png
            ...
        normal/
            ...
        viral_pneumonia/
            ...
    
    Args:
        data_dir (str): Root directory containing class subdirectories
        train (bool): Whether this is for training (True) or validation/testing (False)
        transform (callable, optional): Optional transform to be applied to images
        target_transform (callable, optional): Optional transform to be applied to labels
        device (str): Device to load the data to ("cpu", "cuda:0", etc.)
    """
    def __init__(
        self, data_dir, train=True, transform=None, target_transform=None, device="cpu"
    ):
        # Class mapping: category name -> numeric label
        classes = {"covid": 0, "lung_opacity": 1, "normal": 2, "viral_pneumonia": 3}
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        # Build dataset by collecting all image paths and their corresponding classes
        dataset = []
        for k, v in classes.items():
            imgs = [(k, img) for img in os.listdir(os.path.join(self.data_dir, k))]
            dataset += imgs

        # Shuffle dataset for validation/testing to ensure random sampling
        if not self.train:
            random.shuffle(dataset)

        self.dataset = dataset
        self.labels = [classes[i[0]] for i in dataset]

        assert len(self.dataset) == len(self.labels), "Dataset and labels length mismatch"

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        This method loads an image from disk, applies transformations if specified,
        and returns the image tensor along with its class label.
        
        Args:
            index (int): Index of the sample to fetch
            
        Returns:
            tuple: (image, label) where image is the transformed image tensor
                  and label is the class index
        """
        # Construct full path to the image
        img_path = os.path.join(
            self.data_dir, self.dataset[index][0], self.dataset[index][1]
        )
        
        # Load image and convert to RGB (ensuring 3 channels)
        data = Image.open(img_path)
        data = data.convert('RGB')
        
        # Get corresponding label
        label = self.labels[index]
        
        # Apply transformations if specified
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
            
        return data, label

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.labels)

    def getitem(self, index):
        """
        Alternative method to get a sample from the dataset.
        
        This is a wrapper around __getitem__ for compatibility purposes.
        
        Args:
            index (int): Index of the sample to fetch
            
        Returns:
            tuple: (image, label) pair
        """
        return self.__getitem__(index)


if __name__ == '__main__':
    # Test code to verify dataset functionality
    from utils.enhance_trans import *
    import pdb
    
    # Create a dataset with gamma enhancement transformation
    train_transform = enhance_trans(size=224, enhan_type='gamma')
    train_data = MyDataset(data_dir='../datasets/covid19/train',
                           train=True,
                           transform=train_transform)
    pdb.set_trace()  # Set breakpoint for interactive debugging
