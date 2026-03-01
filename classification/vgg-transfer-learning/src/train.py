
'''
Main training entry point
'''
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_loaders
from src.engine import train_one_epoch, evaluate
from src.model import build_model

def get_device():
    """
    Detects and returns the best available hardware device for PyTorch.
    Checks for NVIDIA GPU (CUDA), Apple Silicon GPU (MPS), and falls back to CPU.
    
    Returns:
        torch.device
    """
    
    # 1. Check for NVIDIA GPU (Standard for AWS, Colab, PC gaming rigs)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Print the exact name of the GPU (e.g., "Tesla T4" or "RTX 4090")
        print(f"✅ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        
    # 2. Check for Apple Silicon GPU (Mac M1, M2, M3, M4)
    # MPS stands for Metal Performance Shaders
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Using Apple Silicon Metal (MPS) GPU")
        
    # 3. Fallback to CPU
    # This will trigger for Intel Macs, or PCs without a dedicated GPU
    else:
        device = torch.device("cpu")
        print("⚠️ No GPU detected. Using CPU.")
        
    return device


def run_training(augment=False):
    device = get_device()
    #Load dataset
    train_loader, val_loader, num_classes = get_loaders(
        data_root= "../../data/Food-5K",
        batch_size=128,
        image_size=200,
        augment=augment
    )

    #Build model
    model = build_model(num_classes=num_classes, freeze_backbone=True)
    model = model.to(get_device())

    #CrossEntropy combines LogSoftmax & NLLLoss
    criterion = nn.CrossEntropyLoss()

    #Only optimize parameters that require gradients
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = 1e-3
    )

    for epoch in range(10):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader, 
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print("-" * 50)


if __name__ == "__main__":

    print("===== Training WITHOUT augmentation =====")
    run_training(augment=False)

    print("\n===== Training WITH augmentation =====")
    run_training(augment=True)
