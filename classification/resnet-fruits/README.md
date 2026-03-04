```markdown
# 🍎 ResNet50 Transfer Learning: Fruits-360 Classification

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)

A modular, production-ready PyTorch pipeline for multi-class fruit classification using a pretrained ResNet50 backbone. This project emphasizes manual training loops, YAML-driven configuration, and dynamic hardware acceleration without relying on high-level wrappers.

---

## 🚀 Project Overview

* **Dataset:** [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits) (60 targeted classes)
* **Backbone:** ResNet50 (ImageNet pretrained weights)
* **Training Strategy:**
  1. **Stage 1:** Freeze the backbone and train the custom classifier head.
  2. **Stage 2 (Optional):** Unfreeze specific layers for fine-tuning.

To match the dataset, the final fully connected layer of the standard ResNet50 is dynamically replaced:

```python
# Replace the final layer for 60-class output
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

```

> **Note:** No Softmax layer is added to the model output because PyTorch's `CrossEntropyLoss` expects raw, unnormalized logits for numerical stability.

---

## 🧠 Architecture Flow

1. **Input:** `3 × 100 × 100` RGB Images
2. **Convolutional Layers:** Low-level feature extraction
3. **Residual Blocks:** Deeper feature extraction via skip connections
4. **Global Average Pooling:** Spatial dimensionality reduction `(2048 × 1 × 1)`
5. **Fully Connected Layer:** Classification output `(2048 → 60)`

---

## 📂 Project Structure

```text
resnet-fruits/
├── configs/
│   └── train.yaml
├── src/
│   ├── dataset.py      # Custom Dataset and DataLoader logic
│   ├── engine.py       # Core training and validation loops
│   ├── metrics.py      # Accuracy and Confusion Matrix calculations
│   ├── model.py        # ResNet50 initialization and modification
│   └── train.py        # Main execution script
├── outputs/
│   └── checkpoints/    # Saved .pth model weights
└── README.md

```

---

## ⚙️ Configuration

Hyperparameters are isolated from the codebase using YAML. This prevents hardcoding and allows for rapid experimentation.

**`configs/train.yaml`**

```yaml
data:
  root: "../../data/fruits-360"
  image_size: 100
  batch_size: 128
  augment: true

model:
  num_classes: 60
  freeze_backbone: true

training:
  epochs: 16
  learning_rate: 0.001
  checkpoint_dir: "../outputs/checkpoints"

```

---

## 🔄 Training Pipeline

The custom `engine.py` handles the complete lifecycle of the model:

* **Data Augmentation:** `RandomRotation` and `HorizontalFlip` to generalize the model and prevent overfitting.
* **Normalization:** Standard ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
* **Manual Loops:** Explicit forward pass, loss calculation, backpropagation, and optimizer stepping.
* **Validation:** Evaluating model performance on unseen data after every epoch.
* **Checkpointing:** Automatic saving of the best-performing model state.

---

## 💻 How to Run

**1. Install dependencies:**

```bash
pip install -r requirements.txt

```

**2. Execute the training script:**

```bash
# Run as a module from the root directory
python -m src.train

```

### ⚡ Device Management

Hardware acceleration is fully automated. The script detects and utilizes the optimal hardware available:

* **CUDA** (NVIDIA GPUs)
* **MPS** (Apple Silicon)
* **CPU** (Fallback)

---

## 📊 Evaluation & Results

The training loop provides real-time terminal feedback for loss and accuracy metrics:

```text
Epoch 1/16
Train Loss: 0.8421 | Train Acc: 0.7653
Val Loss:   0.4328 | Val Acc:   0.9012

```

Upon completion, a **Confusion Matrix** is computed to evaluate class-wise performance and identify edge-case misclassifications. The best model weights are automatically saved to `outputs/checkpoints/best_model.pth`.

