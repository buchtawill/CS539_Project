import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from GrayscaleImagePair import GrayscaleImagePair


class Colorizer(nn.Module):
    def __init__(self):
        super(Colorizer, self).__init__()
        
        self.activation = F.relu
        
        # Encoder (downsampling)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        
        # Decoder (upsampling)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Encoder
        x = self.activation(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.activation(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.activation(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.activation(self.conv4(x))
        x = F.max_pool2d(x, 2)
        
        # Decoder
        x = self.activation(self.deconv1(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.activation(self.deconv2(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.activation(self.deconv3(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.sigmoid(self.deconv4(x))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        return x
        
        
def plot_imgs(epoch):
    # Plot example before and after images
    model.eval()
    with torch.no_grad():
        examples = next(iter(dataloader))[0]  # Take one batch of examples
        examples = examples.to(device)
        reconstructions = model(examples)
        reconstructions = reconstructions.cpu().detach()
        examples = examples.cpu()

        # Plot original and reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
        for i in range(5):
            axes[0, i].imshow(examples[i].permute(1, 2, 0), cmap='gray')
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")
            axes[1, i].imshow(reconstructions[i].permute(1, 2, 0))
            axes[1, i].set_title(f"Reconstructed (epoch={epoch})")
            axes[1, i].axis("off")
        plt.tight_layout()
        plt.savefig(f'./output/epoch_results/epoch{epoch}.png')
        plt.close()
        
if __name__ == '__main__':
    image_dir = 'C:\\Users\\bucht\\OneDrive - Worcester Polytechnic Institute (wpi.edu)\\MQP\\ML\\mario_data\\data\\640_16x9_1000'
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")
    
    #load dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    seed = 42  # Set the seed for reproducibility
    torch.manual_seed(seed)
    full_dataset = GrayscaleImagePair(image_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
       
    '''print(len(train_dataset))
    print(len(train_dataset[0]))
    print(len(train_dataset[0][0])) #grayscale image
    print(len(train_dataset[0][1])) #corresponding color image'''
    # Loop over the train dataset
    #for data in train_dataset:
        # Access the data and perform operations
        # Example: 
        # inputs, targets = data
        # perform training operations
    #gray  = train_dataset[0][0].detach()
    #color = train_dataset[0][1].detach()
    
    #plt.imshow(color.permute(1, 2, 0))
    #plt.show()
    #plt.imshow(gray.permute(1, 2, 0), cmap='gray')
    #plt.show()
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
    print(f"Total train samples: {len(train_dataset)}")
    print(f'Num batches: {len(dataloader)}')
    
    model = Colorizer().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    print("Starting training")
    
    losses = []
    num_epochs = 500
    for epoch in tqdm(range(num_epochs)):
    #for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(dataloader, leave=False):
        #for batch in dataloader:
            in_grays, color_truth = batch
            #print("Sending to device")
            in_grays = in_grays.to(device)
            color_truth = color_truth.to(device)
            
            optimizer.zero_grad()
            
            #print("Modeling inputs")
            color_pred = model(in_grays)
            #print("Calculating loss")
            loss = criterion(color_pred, color_truth)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        scheduler.step(epoch_loss)
        if(epoch % 5 == 0):
            plot_imgs(epoch)
        
        plt.cla()
        plt.clf()
        plt.plot(losses)
        plt.title("Loss vs Epoch")
        plt.ylabel("Loss (MSE)")
        plt.xlabel("Epoch")
        plt.savefig(f'./output/epoch_results/loss.png')
        plt.close()