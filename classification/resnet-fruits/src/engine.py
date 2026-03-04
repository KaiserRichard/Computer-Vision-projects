'''
Implement manual training loop
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device ):

    '''
    Performs ONE full training pass over dataset
    '''    

    # Enable training mode:
    # - Activate Dropout
    # - BatchNorm updates running statistics
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images) # shape : (B, num_classes)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Parameters update
        optimizer.step()
        running_loss += loss.item()

        # Compute predictions
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device ):
    '''
    Evaluation loop
    No gradient updates
    '''

    # Evaluation mode: 
    # - Disable Dropout
    # - Using batchnorm stats
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient tracking to save memory
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs,labels)

            running_loss += loss.item()

            _,preds = torch.max(outputs, dim=1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / len(loader)
        accuracy = correct / total

        return avg_loss, accuracy
