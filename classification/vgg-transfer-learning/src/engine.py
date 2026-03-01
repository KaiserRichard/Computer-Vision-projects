# Training and Evaluation engine for VGG
# This replaces TensorFlow's model.fit()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader



# One full pass over training dataset
def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module,
        device: torch.device):

    # model.train():
    #   - Does not train the model
    #   - Switches certain layers (Dropoout, BatchNorm) into training behavior mode
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in loader:
        #Move data to device
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        #Forward pass:
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backpropagation:
        loss.backward()

        #Update parameters using optimizer
        optimizer.step()

        running_loss += loss.item()

        #Compute predictions
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += (preds==labels).sum().item()
        total_samples += labels.size(0)

    return running_loss / len(loader), correct_predictions / total_samples


def evaluate(model, loader, criterion, device):
    # No gradient computation
    model.eval() # Disable dropout, using batchnorm stats

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            #Forward pass:
            
            #Output tensor shape : (batch_size, num_classes)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            #Compute predictions
            _, preds = torch.max(outputs, 1) #returns (Actual maximum values, predicted labels)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0) #How many items are currently in that specific batch

    return running_loss / len(loader), correct_predictions / total_samples
        



