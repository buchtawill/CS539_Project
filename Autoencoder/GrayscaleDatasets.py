import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class GrayscaleImagePair(Dataset):
    """Load the image, transform it to a tensor when calling __getitem__. This will be slower than using the tensor version, but will use less memory.
    """
    
    def __init__(self, path_to_dataset:str, transform=None):
        self.main_dir = path_to_dataset
        self.transform = transform
        
        self.grayscale_images_path = os.path.join(self.main_dir, 'black')
        self.color_images_path = os.path.join(self.main_dir, 'color')
        
        # Check that the directories exist
        if(not os.path.exists(self.grayscale_images_path)):
            raise Exception(f"Directory {self.grayscale_images_path} does not exist.")

        if(not os.path.exists(self.color_images_path)):
            raise Exception(f"Directory {self.color_images_path} does not exist.")
        
        # Get the filenames from the color images directory and check that the corresponding grayscale images exist
        self.filenames = np.array(os.listdir(self.color_images_path), dtype=str)
        
        #first_color_shape = np.array(Image.open(os.path.join(self.color_images_path, self.filenames[0]))).shape
        #first_gray_shape = np.array(Image.open(os.path.join(self.grayscale_images_path, self.filenames[0])).convert('L')).shape
        
        for name in self.filenames:
            path_to_color = os.path.join(self.color_images_path, name)
            path_to_grayscale = os.path.join(self.grayscale_images_path, name)
            
            if(not os.path.exists(path_to_grayscale) and os.path.exists(path_to_color)):
                raise Exception(f"The file {os.path.join(self.color_images_path, name)} exists but {os.path.join(self.grayscale_images_path, name)} does not.")

            if(not os.path.exists(path_to_color) and os.path.exists(path_to_grayscale)):
                raise Exception(f"The file {path_to_grayscale} exists but {path_to_color} does not.")
            
            # Check that the shapes of the images are the same (confirmed - all images 400x400 pixels)
            # color_shape = np.array(Image.open(path_to_color)).shape
            # gray_shape  = np.array(Image.open(path_to_grayscale).convert('L')).shape
            # print(name)
            #  if(color_shape != first_color_shape):
            #      raise Exception(f"Image {path_to_color} has shape {color_shape} but the first image has shape {first_color_shape}.")
            #  if(gray_shape != first_gray_shape):
            #      raise Exception(f"Image {path_to_grayscale} has shape {gray_shape} but the first image has shape {first_gray_shape}.")
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        color_image_path = os.path.join(self.color_images_path, self.filenames[idx])
        grayscale_image_path = os.path.join(self.grayscale_images_path, self.filenames[idx])
        
        # Load images
        # Without converting, PIL will open the file as a 3-channel image, so we need to convert to gray 
        color_image = Image.open(color_image_path)
        grayscale_image = Image.open(grayscale_image_path).convert('L') 
        
        if self.transform:
            color_image = self.transform(color_image)
            grayscale_image = self.transform(grayscale_image)
        
        return grayscale_image, color_image



class GrayscaleTensorPair(Dataset):
    """This will load tensor representations of the images. Using tensors will speed up the training process, but will require more memory.
    
    """
    def __init__(self, path_to_tensors:str):
        self.main_dir = path_to_tensors
        
        self.grayscale_tensors = torch.load(os.path.join(self.main_dir, 'bw_tensors.pt'))
        self.color_tensors = torch.load(os.path.join(self.main_dir, 'color_tensors.pt'))
        
        if(len(self.grayscale_tensors) != len(self.color_tensors)):
            raise Exception("The number of grayscale and color tensors must be the same.")
             
          
    def __len__(self):
        return len(self.grayscale_tensors)

    def __getitem__(self, idx):
        return self.grayscale_tensors[idx], self.color_tensors[idx] 
    