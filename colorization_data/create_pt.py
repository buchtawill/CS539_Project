import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

image_dir = "./color"

color_tensors_list = []
bw_tensors_list = []

gray_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

color_transform = transforms.Compose([
    transforms.ToTensor()
])

for filename in tqdm(os.listdir(image_dir)):
    color_image = Image.open(os.path.join(image_dir, filename))

    color_tensor = color_transform(color_image)
    bw_tensor = gray_transform(color_image)
    
    color_tensors_list.append(color_tensor)
    bw_tensors_list.append(bw_tensor)
    
color_tensors = torch.stack(color_tensors_list)
bw_tensors = torch.stack(bw_tensors_list)


print(color_tensors.shape)
print(bw_tensors.shape)

torch.save(color_tensors, "./color_tensors.pt")
torch.save(bw_tensors, "./bw_tensors.pt")