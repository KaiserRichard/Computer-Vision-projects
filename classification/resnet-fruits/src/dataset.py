''''
Handles:
- Data augmentation
- ImagFolder dataset
- DataLoader creation
'''

import os 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ImageNet normalizeation values
# ResNet50 was pretrained on ImageNet.
# If we do Not normalize with these statistics.
# feature distribution will mismatch pretrained weights.
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Return a torchvision transform pipeline
def get_transforms(image_size=100, augment=False):

    if augment:
        # Data augmentation only applied during training.
        # These operations are differentiable transformations
        # applied BEFORE converting to tensor.
        data_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            
            #Random horizontal flip (mirror)
            transforms.RandomHorizontalFlip(),

            #Random small rotation
            transforms.RandomRotation(10),

            #Convert PIL image -> PyTorch Tensor
            #Change shape from (H,W,C) to (C,H,W)
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
        ])

    else:
        #Validation transform must NOT use augmentation
        data_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
        ])

    #Output tensor shape after ToTensor(): (3, image_size, image_size)
    return data_transforms

# Build Pytorch DataLoader
def get_loader(data_root, batch_size=128, image_size=100, augment=True):
    train_dir = os.path.join(data_root, "Training")
    val_dir = os.path.join(data_root, "Validation")

    train_dataset = datasets.ImageFolder(
        train_dir, 
        transform=get_transforms(image_size, augment)
    )

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=get_transforms(image_size, False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True # Shuffle only training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, len(train_dataset.classes), train_dataset.classes

