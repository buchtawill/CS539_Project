import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
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
#from GrayscaleDatasets import GrayscaleImagePair
#from gan_colorizer import Generator, Discriminator

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

NUM_EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.0002  
BETA1 = 0.5

def tensor_to_image(tensor: torch.Tensor) -> Image:
    return transforms.ToPILImage()(tensor)

def plot_images(grays, colorizeds, truths, epoch):
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
    #plt.show()
    plt.savefig(f'epoch_results/epoch{epoch}.jpg')
    plt.close()
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO [colorizer.py] Using device: {device} [torch version: {torch.__version__}]')

    #transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])  
    print(f'INFO [colorizer.py] Loading grayscale tensor pair dataset')
    full_dataset = GrayscaleTensorPair('../colorization_data/tensors')
    
    train_size = int(0.8 * len(full_dataset)) #Could be the cause of the issue
    test_size = len(full_dataset) - train_size #####
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator())
    print(f'INFO [colorizer.py] Num of training samples: {len(train_dataset)}')

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f'INFO [colorizer.py] Num batches: {len(dataloader)}')

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss() 
    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    scaler = GradScaler()  

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(NUM_EPOCHS):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for i, (grayscale, real_color) in progress_bar:
            grayscale, real_color = grayscale.to(device), real_color.to(device)

            # Update Discriminator
            discriminator.zero_grad()
            with autocast():
                output = discriminator(real_color).view(-1)
                real_labels = torch.full((output.size(0),), real_label, device=device, dtype=torch.float)
                loss_d_real = criterion(output, real_labels)
                
                fake_color = generator(grayscale)
                output = discriminator(fake_color.detach()).view(-1)
                fake_labels = torch.full((output.size(0),), fake_label, device=device, dtype=torch.float)
                loss_d_fake = criterion(output, fake_labels)
                
            scaler.scale(loss_d_real + loss_d_fake).backward()
            scaler.step(optimizer_d)
            scaler.update()

            # Update Generator
            generator.zero_grad()
            with autocast():
                output = discriminator(fake_color).view(-1)
                generator_labels = torch.full((output.size(0),), real_label, device=device, dtype=torch.float)
                loss_g = criterion(output, generator_labels)
                
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scaler.update()

            epoch_loss_d += (loss_d_real + loss_d_fake).item()
            epoch_loss_g += loss_g.item()

            progress_bar.set_postfix(loss_d=(epoch_loss_d / (i+1)), loss_g=(epoch_loss_g / (i+1)))

        if epoch % 5 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] Loss D: {epoch_loss_d/len(dataloader)}, loss G: {epoch_loss_g/len(dataloader)}')
            plot_images(grayscale, fake_color, real_color, epoch)
    
    torch.save(generator.state_dict(), 'generator_state_dict_500e.pt')
    torch.save(discriminator.state_dict(), 'discriminator_state_dict_500e.pt')
