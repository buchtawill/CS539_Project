import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from GrayscaleDatasets import GrayscaleTensorPair
from GrayscaleDatasets import GrayscaleImagePair

NUM_EPOCHS = 400
BATCH_SIZE = 4

# https://xiangyutang2.github.io/auto-colorization-autoencoders/
# http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf
# ChatGPT used to help translate some features of the model to pytorch (from keras)

class ColorizationAutoencoder(nn.Module):
    def __init__(self):
        super(ColorizationAutoencoder, self).__init__()
        
        # Feature extractor model (start with vanilla autoencoder only)
        # self.feature_extractor = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(feature_size, 1024),
        #     nn.ReLU(inplace=True)
        # )
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        
    def forward(self, x):
        # Feature extractor
        #image_feature = self.feature_extractor(inputs1)
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x

def tensor_to_image(tensor:torch.tensor) -> Image:
    return transforms.ToPILImage()(tensor)

def plot_images(grays, colorizeds, truths, title):
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))
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
    plt.tight_layout()
    plt.savefig(title)
    plt.close()

def sec_to_human(seconds):
    """Return a number of seconds to hours, minutes, and seconds"""
    seconds = seconds % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hours, minutes, seconds)

if __name__ == '__main__':
    tstart = time.time()
    print(f"INFO [colorizer.py] Starting script at {tstart}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO [colorizer.py] Using device: {device} [torch version: {torch.__version__}]')
    print(f'INFO [colorizer.py] Python version: {sys.version_info}')
    model = ColorizationAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #print(model)
    
    #seed = 50  # Set the seed for reproducibility
    #torch.manual_seed(seed)
    print("INFO [colorizer.py] Loading Tensor pair dataset")
    full_dataset = GrayscaleTensorPair('../colorization_data/tensors')
    
    #transform = transforms.Compose([transforms.ToTensor()])
    #full_dataset = GrayscaleImagePair("../colorization_data/images", transform=transform)

    #Create train and test datasets. Set small train set for faster training
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator())
    print(f'INFO [colorizer.py] Num of training samples: {len(train_dataset)}')

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f'INFO [colorizer.py] Num batches: {len(dataloader)}')
    
    #print(len(dataloader.dataset)) # dataloader.dataset: tuples of (grayscale, color)
    #print(dataloader.dataset[0][0].shape) #grayscale image
    #losses_per_epoch = []

    #for epoch in tqdm(range(NUM_EPOCHS)):
    #    for batch in tqdm(dataloader, leave=False):
    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            
            in_grays, color_truths = batch 
            
            in_grays = in_grays.to(device)
            color_truths = color_truths.to(device)
            
            optimizer.zero_grad()
            
            color_preds = model(in_grays)
            loss = criterion(color_preds, color_truths)
            #losses_per_epoch.append(loss)
            loss.backward()
            optimizer.step()
            
            # color_preds = color_preds.cpu()
            # in_grays = in_grays.cpu()
            # color_truths = color_truths.cpu()   
        
        if(epoch % 10 == 0):
            in_grays, color_truths = next(iter(dataloader)) #get first images
            in_grays = in_grays.to(device)
            color_truths = color_truths.to(device)
            color_preds = model(in_grays)
            loss = criterion(color_preds, color_truths)
            #print(f'Epoch {epoch} loss: {loss.item()}')            
            plot_images(in_grays, color_preds, color_truths, f"epoch_results/epoch{epoch}.png")  
            
    torch.save(model.state_dict, './vanilla_encoder_400E.pt')
    
    tEnd = time.time()
    print(f"INFO [colorizer.py] Ending script. Took {tEnd-tstart} seconds.")
    print(f"INFO [colorizer.py] {sec_to_human(tEnd-tstart)}")