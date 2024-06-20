import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GrayscaleImagePair(Dataset):
    def __init__(self, path_to_images, transform=None):
        self.dir_with_images = path_to_images
        self.transform = transform
        
        # List of filenames present in the low-res directory
        self.filenames = [f for f in os.listdir(self.dir_with_images) if os.path.isfile(os.path.join(self.dir_with_images, f))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        color_image_path = os.path.join(self.dir_with_images, self.filenames[idx])
        
        # Load images
        color_image = Image.open(color_image_path).convert('RGB')
        grayscale_image = color_image.convert('L')
        
        if self.transform:
            color_image = self.transform(color_image)
            grayscale_image = self.transform(grayscale_image)
        
        return grayscale_image, color_image
