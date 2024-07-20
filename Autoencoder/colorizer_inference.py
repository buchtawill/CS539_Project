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
from colorizer import ColorizationAutoencoder
from GrayscaleDatasets import GrayscaleImagePair
from GrayscaleDatasets import GrayscaleTensorPair


def tensor_to_image(tensor:torch.tensor) -> Image:
    return transforms.ToPILImage()(tensor)

def save_6_images(grays, predictions, colors, title='results.png') -> None:
    fig, axs = plt.subplots(3, 5, figsize=(12, 8))
    for i in range(5):
        axs[0, i].imshow(tensor_to_image(grays[i].cpu()), cmap='gray')
        axs[0, i].axis('off')
        axs[0, i].set_title('Grayscale')
        
        axs[1, i].imshow(tensor_to_image(predictions[i].cpu()))
        axs[1, i].axis('off')
        axs[1, i].set_title('Colorized')
        
        axs[2, i].imshow(tensor_to_image(colors[i].cpu()))
        axs[2, i].axis('off')
        axs[2, i].set_title('Truth')
    plt.tight_layout()
    #plt.savefig(title)
    plt.show()
    plt.close()

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO [colorizer_inference.py] Using device: {device} [torch version: {torch.__version__}]')
    print(f'INFO [colorizer_inference.py] Python version: {sys.version_info}')
    
    model = ColorizationAutoencoder().to(device)
    #Since state dict was saved as a function call and not just weights, need to "call" the function
    checkpoint = torch.load('./vanilla_1kE_8bat.pt')() 
    #print(checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    criterion = nn.MSELoss()
    
    seed = 50  # Set the seed for reproducibility
    torch.manual_seed(seed)
    print("INFO [colorizer_inference.py] Loading Tensor pair dataset")
    full_dataset = GrayscaleTensorPair('../colorization_data/tensors')
    
    # Create train and test datasets. Set small train set for faster training
    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator())
    num_test_samples = len(test_dataset)
    print(f'INFO [colorizer_inference.py] Num of test samples: {num_test_samples}')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=6)
    
    #Store the first 6 images, then calculate total loss
    with torch.no_grad():
        running_loss = 0.0
        first_batch = True
        for batch in tqdm(test_dataloader):
            in_grays, color_truths = batch
            in_grays = in_grays.to(device)
            color_truths = color_truths.to(device)
            
            color_preds = model(in_grays)
            loss = criterion(color_preds, color_truths)
            running_loss += loss.item()
            
            if(first_batch):
                first_batch = False
                save_6_images(grays=in_grays, predictions=color_preds, colors=color_truths, title='results.png')
            #tb_writer.add_scalar("Loss/test", loss, i)
        print(f"Average loss of test set: {running_loss / num_test_samples}")