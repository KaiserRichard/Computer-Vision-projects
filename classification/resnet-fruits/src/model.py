''''
Define ResNot50 Transfer Learning model
'''

import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=60, freeze_backbone=True):
    '''
    Load pretrained ResNet50 and replace classifier head
    '''
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        # Freeze convolutional backbone only
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully connected layer
    # Original: 2048 -> 1000
    # New: 2048 -> num_classes = 60
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # DO NOT add Softmax here
    # Becasue CrossEntropyLoss expects raw logits

    return model


