# Model definition for Transfer Learning using VGG16
import torch.nn as nn
import torchvision.models as models

#Loads pretrained VGG16 and replaces final classifier layer.
def build_model(num_classes=2, freeze_backbone=True):
    #Load pretrained VGG16 (trained on ImageNet)
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    #VGG architecture:
    # Image -> VGG feature extractor -> Feature vector -> Then add own classifier & Train only that

    if freeze_backbone:
        '''
        We freeze convolutional layeres to:
            - Preserve pretained feature extractor
            - Reduce training time
            - Prevent overfitting on small dataset
        '''
        for param in model.features.parameters():
            param.requires_grad = False
    
    #Replace final fully connected layer:
    #Original: 4096 -> 1000 (ImageNet classes)
    #New: 4096 -> num_classes (2)

    #Look at layer number 6 => 4096 input connections from the previous layer
    #From index 0 -> 5 remains with its pre-trained knowledge, and only layer 6 will learn
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    #IMPORTANT:
    #DO NOT add Softmax here
    #Because CrossEntropyLoss expects raw logits

    return model