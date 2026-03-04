"""
Main entry point

Responsibilities:
- Load configuration
- Select compute device
- Build dataset and model
- Run training loop
- Save best checkpoint
- Compute confusion matrix
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_loader
from src.model import build_model
from src.engine import train_one_epoch, evaluate
from src.metrics import compute_confusion_matrix


# -------------------------------------------------
# Load YAML configuration file
# -------------------------------------------------
def load_config(config_path="configs/train.yaml"):
    """
    Reads hyperparameters and paths from YAML.
    Keeps training script clean and configurable.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# -------------------------------------------------
# Device selection
# -------------------------------------------------
def get_device():
    """
    Detects best available hardware:
    - CUDA (NVIDIA GPU)
    - MPS (Apple Silicon)
    - CPU fallback
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")

    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")

    return device


# -------------------------------------------------
# Main training pipeline
# -------------------------------------------------
def run_training(config):

    # -------- DEVICE --------
    device = get_device()

    # -------- DATA --------
    # DataLoader returns CPU tensors by default.
    # They will be moved to device inside engine.
    train_loader, val_loader, num_classes, classes = get_loader(
        data_root=config["data"]["root"],
        batch_size=config["data"]["batch_size"],
        image_size=config["data"]["image_size"],
        augment=config["data"]["augment"],
    )

    # -------- MODEL --------
    # Build transfer learning model
    model = build_model(
        num_classes=config["model"]["num_classes"],
        freeze_backbone=config["model"]["freeze_backbone"]
    )
    model = model.to(device)

    # -------- LOSS --------
    # CrossEntropyLoss expects:
    #   logits: (B, C)
    #   labels: (B,)
    criterion = nn.CrossEntropyLoss()

    # -------- OPTIMIZER --------
    # Only update parameters where requires_grad=True
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"]
    )

    # -------- TRAINING LOOP --------
    best_val_loss = float("inf")
    epochs = config["training"]["epochs"]

    # Ensure checkpoint directory exists
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)

    for epoch in range(epochs):

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

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(
                config["training"]["checkpoint_dir"],
                "best_model.pth"
            )
            torch.save(model.state_dict(), save_path)
            print("Saved best model.")

    # -------- CONFUSION MATRIX --------
    cm = compute_confusion_matrix(model, val_loader, device)
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    config = load_config()
    run_training(config)