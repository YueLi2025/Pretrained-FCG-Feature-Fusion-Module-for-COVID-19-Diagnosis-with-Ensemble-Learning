import os
import random
from typing import Optional
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Custom PyTorch dataset for loading medical image data.

    Args:
        data_dir (str): Path to the dataset directory.
        train (bool, optional): If True, dataset is used for training (shuffling disabled for test/val). Defaults to True.
        transform (callable, optional): Transformations applied to the images. Defaults to None.
        target_transform (callable, optional): Transformations applied to the labels. Defaults to None.
        device (str, optional): Device to load the dataset ('cpu' or 'cuda'). Defaults to "cpu".
    """

    def __init__(
        self, 
        data_dir: str, 
        train: bool = True, 
        transform: Optional[callable] = None, 
        target_transform: Optional[callable] = None, 
        device: str = "cpu"
    ):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        # Define class labels
        self.classes = {"covid": 0, "lung_opacity": 1, "normal": 2, "viral_pneumonia": 3}

        # Load dataset
        self.dataset = [
            (class_name, img) 
            for class_name, label in self.classes.items()
            for img in os.listdir(os.path.join(self.data_dir, class_name))
        ]
        
        # Shuffle dataset if used for validation/testing
        if not self.train:
            random.shuffle(self.dataset)

        # Extract labels
        self.labels = [self.classes[class_name] for class_name, _ in self.dataset]

        # Ensure dataset and labels match
        assert len(self.dataset) == len(self.labels), "Dataset size and labels do not match!"

    def __getitem__(self, index: int):
        """Retrieve an image and its corresponding label.

        Args:
            index (int): Index of the item.

        Returns:
            Tuple[Tensor, int]: Transformed image and label.
        """
        class_name, img_name = self.dataset[index]
        img_path = os.path.join(self.data_dir, class_name, img_name)

        # Load image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        label = self.labels[index]

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.labels)
