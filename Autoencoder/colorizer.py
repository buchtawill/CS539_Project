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
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from GrayscaleDatasets import GrayscaleImagePair
from GrayscaleDatasets import GrayscaleTensorPair
from torch.utils.tensorboard import SummaryWriter

NUM_EPOCHS = 1000
BATCH_SIZE = 8

# https://xiangyutang2.github.io/auto-colorization-autoencoders/
# http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf
# ChatGPT used to help translate some features of the model to pytorch (from keras)

class ColorizationAutoencoder(nn.Module):
    def __init__(self):
        super(ColorizationAutoencoder, self).__init__()
        
        # Low level features
        self.low_level_features = nn.Sequential(
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
        )
        
        # mid level features
        self.mid_level_features = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Global features network
        self.global_features = nn.Sequential(
            #input: 512x50x50
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            #512x25x25
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            #512x25x25
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            #512x13x13
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            #512x6x6
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            #512x3x3
            nn.Flatten(),
            # 4608x1
            nn.Linear(in_features=4608, out_features=1024),
            nn.Linear(in_features=1024, out_features=512),
            nn.Linear(in_features=512, out_features=256),
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.colorizer = nn.Sequential(
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
        
        x = self.low_level_features(x)              #Output: bat x 512 x 50 x 50
        from_mid_level = self.mid_level_features(x) #Output: bat x 256 x 50 x 50
        from_global = self.global_features(x)       #Output: bat x 256
        from_global = from_global.unsqueeze(-1)     #Output: bat x 256 x 1
        from_global = from_global.unsqueeze(-1)     #Output: bat x 256 x 1 x 1
        
        from_global = from_global.expand(-1, -1, from_mid_level.size(2), from_mid_level.size(3))  # Output: bat x 256 x 50 x 50

        combined_features = torch.cat((from_mid_level, from_global), dim=1)  # Output: bat x 512 x 50 x 50
        x = self.fusion(combined_features)
        
        x = self.colorizer(x)
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
    #plt.savefig(title)
    plt.show()
    plt.close()


def train_with_tqdm(model, dataloader, optimizer, tb_writer, scheduler, criterion, nSamples):
    for epoch in tqdm(range(NUM_EPOCHS)):
        for batch in tqdm(dataloader, leave=False):
            
            in_grays, color_truths = batch 
            
            in_grays = in_grays.to(device)
            color_truths = color_truths.to(device)
            
            optimizer.zero_grad()
            
            color_preds = model(in_grays)
            loss = criterion(color_preds, color_truths)
            tb_writer.add_scalar("Loss/train", loss, epoch)
            
            loss.backward()
            optimizer.step()
            #scheduler.step()
        
        if(epoch % 1 == 0):
            in_grays, color_truths = next(iter(dataloader)) #get first images
            in_grays = in_grays.to(device)
            color_truths = color_truths.to(device)
            color_preds = model(in_grays)
            loss = criterion(color_preds, color_truths)
            #print(f'Epoch {epoch} loss: {loss.item()}')            
            plot_images(in_grays, color_preds, color_truths, f"epoch_results/epoch{epoch}.png")
            
def train_normal(model, dataloader, optimizer, tb_writer, scheduler, criterion, nSamples):
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for batch in dataloader:
            
            in_grays, color_truths = batch 
            
            in_grays = in_grays.to(device)
            color_truths = color_truths.to(device)
            
            optimizer.zero_grad()
            
            color_preds = model(in_grays)
            loss = criterion(color_preds, color_truths)
            
            loss.backward()
            optimizer.step() 
            #scheduler.step()
            running_loss += loss.item()
        
        running_loss /= float(nSamples)
        tb_writer.add_scalar("Loss/train", running_loss, epoch)
        print(f'Epoch {epoch:>{6}}\t loss: {running_loss:.8f}')
        
        if(epoch % 10 == 0):
            in_grays, color_truths = next(iter(dataloader)) #get first images
            in_grays = in_grays.to(device)
            color_truths = color_truths.to(device)
            color_preds = model(in_grays)
            loss = criterion(color_preds, color_truths)
            
            plot_images(in_grays, color_preds, color_truths, f"epoch_results/epoch{epoch}.png")
            

def eval_dataset_normal(model, dataloader, criterion, tb_writer):
    with torch.no_grad:
        for batch in dataloader:
            in_grays, color_truths = batch
            in_grays = in_grays.to(device)
            color_truths = color_truths.to(device)
            
            color_preds = model(in_grays)
            loss = criterion(color_preds, color_truths)
            #tb_writer.add_scalar("Loss/test", loss, i)
            

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
    
    
    #Set up device, model, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO [colorizer.py] Using device: {device} [torch version: {torch.__version__}]')
    print(f'INFO [colorizer.py] Python version: {sys.version_info}')
    model = ColorizationAutoencoder().to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Get dataset
    seed = 50  # Set the seed for reproducibility
    torch.manual_seed(seed)
    print("INFO [colorizer.py] Loading Tensor pair dataset")
    full_dataset = GrayscaleTensorPair('../colorization_data/tensors')
    # print("INFO [colorizer.py] Loading Image pair dataset")
    # transform = transforms.Compose([transforms.ToTensor()])
    # full_dataset = GrayscaleImagePair("../colorization_data/images", transform=transform)
    
    # Create train and test datasets. Set small train set for faster training
    train_size = int(0.25 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator())
    num_train_samples = len(train_dataset)
    print(f'INFO [colorizer.py] Num of training samples: {num_train_samples}')

    # Get Dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader  = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f'INFO [colorizer.py] Num training batches: {len(train_dataloader)}')
    #scheduler = StepLR(optimizer=optimizer, step_size=20, gamma=0.5)
    tb_writer = SummaryWriter()
    
    model.train()
    # train_normal(model=model, dataloader=train_dataloader, optimizer=optimizer, tb_writer=tb_writer, scheduler=None, criterion=criterion, nSamples=num_train_samples)
    train_with_tqdm(model=model, dataloader=train_dataloader, optimizer=optimizer, tb_writer=tb_writer, scheduler=None, criterion=criterion, nSamples=num_train_samples)
            
    #model.eval()
    #eval_dataset_normal(model=model, dataloader=test_dataloader, criterion=criterion, tb_writer=tb_writer)
    
    tb_writer.flush()
    torch.save(model.state_dict(), './vanilla_1kE_4bat_dict.pth')
    
    tEnd = time.time()
    print(f"INFO [colorizer.py] Ending script. Took {tEnd-tstart} seconds.")
    print(f"INFO [colorizer.py] HH:MM:SS --> {sec_to_human(tEnd-tstart)}")