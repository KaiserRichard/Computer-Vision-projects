'''
Evaluation utilities
'''

import torch
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return confusion_matrix(all_labels, all_preds)



