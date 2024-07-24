import gc
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from GrayscaleDatasets import GrayscaleTensorPair

NUM_EPOCHS = 3
BATCH_SIZE = 6

class ColorizationCNN(nn.Module):
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def tensor_to_image(tensor:torch.Tensor) -> Image:
    return transforms.ToPILImage()(tensor)

def plot_images(grays, colorizeds, truths):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        axs[0, i].imshow(tensor_to_image(grays[i].cpu()), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title('Grayscale')
        
        axs[1, i].imshow(tensor_to_image(colorizeds[i].cpu()))
        axs[1, i].axis('off')
        axs[1, i].set_title('Colorized')
        
        axs[2, i].imshow(tensor_to_image(truths[i].cpu()))
        axs[2, i].axis('off')
        axs[2, i].set_title('Truth')
    plt.show()

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO [colorizer.py] Using device: {device} [torch version: {torch.__version__}]')
    model = ColorizationCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # full_dataset = GrayscaleTensorPair(r"C:\Users\zoaib\Downloads\HumzaProject\colorization_data\tensors")
    full_dataset = GrayscaleTensorPair('../colorization_data/tensors')

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator())
    print(f'INFO [colorizer.py] Num of training samples: {len(train_dataset)}')

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f'INFO [colorizer.py] Num batches: {len(dataloader)}')
    
    losses_per_epoch = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        for batch in tqdm(dataloader, leave=False):
            in_grays, color_truths = batch 
            
            in_grays = in_grays.to(device)
            color_truths = color_truths.to(device)
            
            optimizer.zero_grad()
            
            color_preds = model(in_grays)
            loss = criterion(color_preds, color_truths)
            # losses_per_epoch.append(loss)
            loss.backward()
            optimizer.step()
            
            # color_preds = color_preds.cpu()
            # in_grays = in_grays.cpu()
            # color_truths = color_truths.cpu()   
            # gc.collect()
            
        if(epoch % 1 == 0):
            plot_images(in_grays, color_preds, color_truths)