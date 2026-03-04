🍎 ResNet50 Transfer Learning – Fruits Classification
This project implements transfer learning using a pretrained ResNet50 model for multi-class fruit classification on the Fruits-360 dataset.

The entire training pipeline is built in PyTorch using a fully modular architecture with manual training loops (no model.fit()), YAML-based configuration, and automatic device detection.

🚀 Project Overview
Backbone: ResNet50 (ImageNet pretrained)

Dataset: Fruits-360

Number of Classes: 60

Training Strategy:

Stage 1: Freeze backbone, train classifier head.

Optional Stage 2: Fine-tune backbone layers.

The final fully connected layer of the standard ResNet50 is dynamically replaced to match our dataset:

Python
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
Note: No softmax layer is added to the model output because PyTorch's CrossEntropyLoss expects raw, unnormalized logits.

🧠 Architecture Flow
Input: (3 × 100 × 100) RGB Images

Convolutional Layers (Feature extraction)

Residual Blocks (Skip connections)

Global Average Pooling (Spatial dimensionality reduction)

Fully Connected Layer (2048 → 60)

📂 Project Structure
Plaintext
resnet-fruits/
├── configs/
│   └── train.yaml
├── src/
│   ├── dataset.py
│   ├── engine.py
│   ├── metrics.py
│   ├── model.py
│   └── train.py
├── outputs/
│   └── checkpoints/
└── README.md
⚙️ Configuration
All hyperparameters are isolated from the code and defined in configs/train.yaml. This prevents hardcoding and makes experimentation seamless.

YAML
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
🔄 Training Pipeline
The custom training loop handles:

Data Augmentation: RandomRotation, HorizontalFlip to prevent overfitting.

Normalization: Standard ImageNet mean and standard deviation.

Manual Loops: Explicit forward pass, loss calculation, backpropagation, and optimizer steps.

Validation: Evaluating model performance on unseen data after every epoch.

Checkpointing: Automatic saving of the best-performing model state.

💻 How to Run
Install dependencies:

Bash
pip install -r requirements.txt
Run the training script from the project root:

Bash
python -m src.train
Device management is completely automatic! The script will automatically detect and run on:

CUDA (NVIDIA GPUs)

MPS (Apple Silicon Macs)

CPU (Fallback)

📊 Evaluation & Results
During training, the script outputs standard metrics:

Plaintext
Epoch 1/16
Train Loss: 0.8421 | Train Acc: 0.7653
Val Loss: 0.4328 | Val Acc: 0.9012
After training completes, a Confusion Matrix is computed to evaluate class-wise performance and identify edge-case misclassifications.

Best model weights are automatically saved to: outputs/checkpoints/best_model.pth

🎯 Key Learning Points
Transfer learning with pretrained Convolutional Neural Networks (CNNs).

Freezing and unfreezing specific backbone layers for fine-tuning.

Writing manual PyTorch training and validation loops from scratch.

Handling dynamic device management (CUDA / MPS / CPU).

Using YAML for scalable experiment configuration.

Structuring deep learning projects modularly for production.

🛠 Requirements
Python 3.11+

PyTorch

Torchvision

scikit-learn

PyYAML
