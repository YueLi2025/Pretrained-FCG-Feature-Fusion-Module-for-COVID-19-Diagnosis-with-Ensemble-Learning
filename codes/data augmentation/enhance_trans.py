"""
Image Preprocessing with Basic Augmentation and Enhancement
Enhancement options:
- Gamma Correction (Adaptive)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
"""

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from albumentations.augmentations.transforms import CLAHE


class ImageEnhancementTransform:
    """
    Applies image preprocessing with optional enhancement (Gamma Correction or CLAHE).
    
    Args:
        size (int): Target image size (square).
        enhan_type (str, optional): Type of enhancement to apply. Choose from:
            - 'gamma': Adaptive Gamma Correction
            - 'clahe': Contrast Limited Adaptive Histogram Equalization
            - None: No additional enhancement
        show (bool, optional): If True, returns both transformed image and transform object (for visualization).
    """
    
    def __init__(self, size: int, enhan_type: str = None, show: bool = False):
        self.enhan_type = enhan_type
        self.show = show

        # Define base transformations (common to all images)
        self.base_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5264, 0.5264, 0.5276], std=[0.2018, 0.2018, 0.2022]),
        ])

    def __call__(self, img: Image.Image):
        """
        Apply selected enhancement (if any) followed by base transformations.
        
        Args:
            img (PIL.Image.Image): Input image.
        
        Returns:
            Transformed image tensor (if show=False).
            Tuple (transformed image, base transform object) if show=True.
        """
        # Apply enhancement if specified
        if self.enhan_type == "gamma":
            img = self.apply_gamma_correction(img)
        elif self.enhan_type == "clahe":
            img = self.apply_clahe(img)

        transformed_img = self.base_transform(img)

        return (transformed_img, self.base_transform) if self.show else transformed_img

    def apply_gamma_correction(self, img: Image.Image, verbose: bool = False) -> Image.Image:
        """
        Performs adaptive gamma correction to balance brightness.

        Args:
            img (PIL.Image.Image): Input image.
            verbose (bool, optional): If True, prints computed gamma value.
        
        Returns:
            PIL.Image.Image: Gamma-corrected image.
        """
        mid = 0.5  # Desired mid-tone
        mean_intensity = np.mean(img.convert("L"))  # Compute mean intensity in grayscale
        gamma = np.log(mid * 255) / np.log(mean_intensity)  # Adaptive gamma calculation

        if verbose:
            print(f"Gamma Correction Value: {gamma:.4f}")

        return transforms.functional.adjust_gamma(img, gamma=gamma)

    def apply_clahe(self, img: Image.Image) -> Image.Image:
        """
        Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            img (PIL.Image.Image): Input image.

        Returns:
            PIL.Image.Image: Image with enhanced contrast using CLAHE.
        """
        clahe = CLAHE(p=1)  # Ensure transformation is always applied
        enhanced_img = clahe(image=np.array(img))["image"]  # Convert PIL → NumPy → Apply CLAHE
        return Image.fromarray(enhanced_img)  # Convert back to PIL format
