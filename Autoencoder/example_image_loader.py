import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from GrayscaleDatasets import GrayscaleImagePair

NUM_EPOCHS = 500
BATCH_SIZE = 8

def tensor_to_image(tensor:torch.tensor) -> Image:
    return transforms.ToPILImage()(tensor)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    #seed = 50  # Set the seed for reproducibility
    #torch.manual_seed(seed)
    full_dataset = GrayscaleImagePair("../colorization_data/images", transform=transform)

    #Create train and test datasets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=torch.Generator())

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Total train samples: {len(train_dataset)}")
    print(f'Num batches: {len(dataloader)}')

    for epoch in tqdm(range(NUM_EPOCHS)):
        for batch in tqdm(dataloader, leave=False):
            in_grays, color_truths = batch 
            
            gray = in_grays[0]
            color = color_truths[0]
            
            #im = tensor_to_image(gray)
            #im.show()
            
            # Train the model
            
            