"""
Class Activation Mapping (CAM) Generator for COVID-19 Classification

This module implements a heatmap generator that visualizes the regions of interest
in chest X-ray/CT images that contribute most to the model's classification decision.

The implementation uses Class Activation Mapping (CAM) technique to highlight
the discriminative image regions used by the CNN to identify the different
pathologies (COVID-19, lung opacity, normal, viral pneumonia).

References:
    Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016).
    Learning Deep Features for Discriminative Localization.
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2921-2929).
"""

import os
import numpy as np
import time
import sys
from PIL import Image

import cv2  # OpenCV for image processing and visualization

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

# from DensenetModels import DenseNet121
# from DensenetModels import DenseNet169
# from DensenetModels import DenseNet201
from model import MyModel

#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

class HeatmapGenerator():
    """
    Class Activation Mapping (CAM) based heatmap generator.
    
    This class implements the CAM technique to visualize which regions of an image
    are important for the model's classification decision. It extracts the feature maps
    from the last convolutional layer and weights them by the importance of each
    feature map for a specific class, determined by the weights in the final
    classification layer.
    
    The resulting heatmap is overlaid on the original image to show which
    regions contributed most to the diagnosis, providing interpretability
    for the model's decisions.
    """
 
    def __init__(self, pathModel, nnArchitecture, nnClassCount, transCrop):
        """
        Initialize the heatmap generator.
        
        Args:
            pathModel (str): Path to the trained model checkpoint
            nnArchitecture (str): Neural network architecture name (e.g., 'DENSE-NET-121')
            nnClassCount (int): Number of classes (4 for COVID-19 classification)
            transCrop (int): Size to crop/resize the input images (e.g., 224 for DenseNet)
        """
        #---- Initialize the network
        # Load the model architecture
        # Note: The commented code below shows alternative model architectures that could be used
        # if nnArchitecture == 'DENSE-NET-121': model = MyModel(4, 'densenet121').cuda()
        # if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True).cuda()
        # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True).cuda()
        # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True).cuda()
        
        # model = torch.nn.DataParallel(model).cuda()
        model = MyModel(4, 'densenet121').cuda()
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['model'])

        # Extract the feature extractor part of the model (before the classifier)
        # self.model = model.module.densenet121.features
        self.model = model.backbone.features
        self.model.eval()  # Set to evaluation mode
        
        #---- Get the weights from the last layer (used for CAM generation)
        # These weights represent the importance of each feature map for each class
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform pipeline
        # Standard normalization for ImageNet-pretrained models
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate(self, pathImageFile, pathOutputFile, transCrop):
        """
        Generate a CAM heatmap for a given image and save the visualization.
        
        This method:
        1. Loads and preprocesses the input image
        2. Extracts feature maps from the model's last convolutional layer
        3. Weights these feature maps according to their importance for classification
        4. Creates a heatmap visualization
        5. Overlays the heatmap on the original image
        6. Saves the result
        
        Args:
            pathImageFile (str): Path to the input image file
            pathOutputFile (str): Path where the output visualization will be saved
            transCrop (int): Size to crop/resize the input images
        """
        #---- Load image, transform, convert to tensor
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)  # Add batch dimension
        
        input = torch.autograd.Variable(imageData)
        
        # Move model and input to GPU and run forward pass
        self.model.cuda()
        output = self.model(input.cuda())
        
        #---- Generate heatmap by weighting feature maps with class weights
        heatmap = None
        for i in range(0, len(self.weights)):
            map = output[0, i, :, :]  # Extract feature map i
            if i == 0: 
                heatmap = self.weights[i] * map  # Initialize with weighted first map
            else: 
                heatmap += self.weights[i] * map  # Add subsequent weighted maps
        
        #---- Blend original image and heatmap for visualization
        # Convert heatmap to numpy for OpenCV processing
        npHeatmap = heatmap.cpu().data.numpy()

        # Load and resize original image for overlay
        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        # Normalize heatmap to 0-1 range and resize to match original image
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        
        # Apply colormap to heatmap (red = high activation, blue = low)
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        # Overlay heatmap on original image (50% heatmap, 50% original)
        img = heatmap * 0.5 + imgOriginal
            
        # Save the visualization
        cv2.imwrite(pathOutputFile, img)
        
#-------------------------------------------------------------------------------- 

# Example usage and batch processing

# Path to a sample COVID-19 image
pathInputImage = '../datasets/covid19/test/covid/COVID-100.png'
# Output path for the heatmap visualization
pathOutputImage = 'heatmap.png'
# Path to the trained model checkpoint
pathModel = 'checkpoints/densenet121_best.ckpt'

# Model configuration
nnArchitecture = 'DENSE-NET-121'
nnClassCount = 4  # 4 classes: covid, lung_opacity, normal, viral_pneumonia

# Image size for processing
transCrop = 224  # Standard input size for DenseNet121

# Initialize heatmap generator
h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)

# Class names for directory organization
classes = ['covid', 'lung_opacity', 'normal', 'viral_pneumonia']
# Path to test dataset
data_dir = '../datasets/covid19/test/'
# Output directory for heatmaps
out_dir = 'heatmaps'
os.makedirs(out_dir, exist_ok=True)

# Generate heatmaps for all test images, organized by class
for cls in classes:
    # Create output directory for each class
    os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
    # Process each image in the class directory
    for file in os.listdir(os.path.join(data_dir, cls)):
        in_image = os.path.join(data_dir, cls, file)
        out_image = os.path.join(out_dir, cls, file)
        # Generate and save heatmap
        h.generate(in_image, out_image, transCrop)
        print(f'Finish {out_image}')