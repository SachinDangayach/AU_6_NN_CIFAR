# CIFAR-10 CNN with Advanced Convolutions 🚀

A comprehensive PyTorch implementation for CIFAR-10 image classification using advanced convolutional techniques including Depthwise Separable Convolutions, Dilated Convolutions, and Global Average Pooling, enhanced with Albumentations data augmentation.

## 🎯 Project Goals

### Model Architecture Requirements
- **Architecture**: C1C2C3C4O (Convolutional blocks without MaxPooling)
- **Receptive Field**: > 44 pixels
- **Advanced Convolutions**:
  - ✅ Depthwise Separable Convolution
  - ✅ Dilated Convolution 
  - ✅ Global Average Pooling
- **Target Performance**:
  - ✅ Achieve 85% test accuracy
  - ✅ Total parameters < 200k
  - ✅ Strided convolution instead of MaxPooling

### Data Augmentation ✅
- **Albumentations Library**:
  - ✅ HorizontalFlip
  - ✅ ShiftScaleRotate  
  - ✅ CoarseDropout (max_holes=1, max_height=16px, max_width=16px)

## 🏗️ Project Structure

```
AU_7_CIFAR/
├── models/
│   ├── cifar10_net.py         # Model architecture (CIFAR10Net)
│   ├── trainer.py             # Training functions
│   └── tester.py              # Testing functions
├── data/
│   └── dataset.py             # Data loading with Albumentations
├── utils/
│   └── visualization.py       # Visualization and analysis utilities
├── cifar10_trainer.py         # Command-line training script
├── train_cifar10.py           # Legacy training script
├── cifar10_training.ipynb     # Jupyter notebook for experimentation
├── config.py                  # Configuration management
├── infer_cifar10.py           # Model inference script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 📋 Features

### 🧠 Model Architecture
- **CIFAR10Net**: Custom architecture designed for CIFAR-10 classification
- **Depthwise Separable Convolution**: Efficient feature extraction in Conv Block 3
- **Dilated Convolution**: Increased receptive field without parameter overhead
- **Global Average Pooling**: Parameter-efficient classification head
- **No MaxPooling**: Uses strided convolutions for spatial dimension reduction

### 📊 Data Pipeline
- **Albumentations Integration**: Advanced augmentations for better generalization
- **Custom Dataset**: Seamless integration with PyTorch DataLoader
- **Smart Normalization**: CIFAR-10 specific mean/std normalization

### 📈 Visualization & Analysis
- **Training Curves**: Comprehensive loss and accuracy plotting
- **Misclassified Images**: Visual analysis of model errors
- **Model Predictions**: Random sample validation with confidence scores
- **Classification Report**: Detailed performance metrics

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd AU_7_CIFAR
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Training

#### Command-Line Training (Recommended)
```bash
python cifar10_trainer.py
```

#### CPU-Optimized Training
```bash
python cifar10_trainer.py --epochs 30 --batch_size 64 --lr 0.1
```

#### GPU-Optimized Training
```bash
python cifar10_trainer.py --epochs 30 --batch_size 256 --lr 0.1 --output_dir runs/exp1
```

#### Legacy Training Script
```bash
python train_cifar10.py --epochs 20 --batch_size 128
```

#### Jupyter Notebook Option
```bash
jupyter notebook cifar10_training.ipynb
```

#### Command Line Arguments
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 128)
- `--lr`: Learning rate (default: 0.1)
- `--momentum`: SGD momentum (default: 0.9)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--dropout`: Dropout probability (default: 0.1)
- `--output_dir`: Output directory for models and plots (default: "outputs")
- `--show_model_summary`: Display model architecture summary

### Usage Examples

```python
# Quick training with default settings
python train_cifar10.py

# Extended training for better accuracy
python train_cifar10.py --epochs 50 --batch_size 128

# GPU-optimized training
python train_cifar10.py --batch_size 256 --num_workers 4

# Experiment with different hyperparameters
python train_cifar10.py --lr 0.05 --momentum 0.95 --weight_decay 5e-4
```

## 🔧 Model Architecture Details

### CIFAR10Net Specifications

```python
# Model Parameters: < 200,000
# Receptive Field: > 44 pixels
# Target Accuracy: > 85%

Architecture Overview:
C1 (Conv Block 1): 3→32→32 channels, 32×32→32×32
C2 (Conv Block 2): 32→64→64 channels, 32×32→32×32  
C3 (Conv Block 3): 64→64→128 channels (DWS + Dilated)
C4 (Conv Block 4): 128→256→512 channels, 32×32→16×16 (stride=2)
GAP + FC: 512→10 classes
```

### Advanced Convolutional Techniques

1. **Depthwise Separable Convolution**:
   - Separates spatial and channel-wise filtering
   - Reduces parameters while maintaining feature extraction capability
   - Applied in Conv Block 3

2. **Dilated Convolution**:
   - Increases receptive field without additional parameters
   - Effective for multi-scale feature extraction
   - Applied in Conv Block 3

3. **Global Average Pooling**:
   - Replaces fully connected layers
   - Reduces overfitting and parameters
   - Maintains spatial importance

## 📊 Training Results

### Performance Metrics
- **Test Accuracy**: 85%+ ✅
- **Training Accuracy**: 87%+
- **Parameters**: < 200k ✅
- **Receptive Field**: > 44 ✅

### Training Improvements
- **Data Augmentation**: Reduced train-test gap
- **Regularization**: Dropout + Weight Decay
- **Learning Rate Scheduling**: Step decay for fine-tuning

## 🔍 Analysis & Visualization

### Training Curves
Automatically generated plots showing:
- Training/Test Loss over epochs
- Training/Test Accuracy over epochs
- Final performance metrics

### Model Validation
- **Random Sample Predictions**: Visual validation of model accuracy
- **Misclassified Images**: Analysis of prediction errors
- **Classification Report**: Per-class performance metrics

### Example Analysis Commands
```python
# Load trained model
model_path = "outputs/cifar10_model.pth"
checkpoint = torch.load(model_path)
model = checkpoint['model_class']()
model.load_state_dict(checkpoint['model_state_dict'])

# Generate classification_report
report = generate_classification_report(model, test_loader, classes)

# Show misclassified images
show_misclassified_images(model, classes, test_loader, num_of_images=20)
```

## 🛠️ Customization

### Model Architecture
```python
# Modify model parameters
model = CIFAR10Net(dropout_value=0.2, num_classes=10)

# Count parameters
print(f"Model parameters: {model.count_parameters():,}")
```

### Data Augmentation
```python
# Custom Albumentations pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5),
    A.CoarseDropout(max_holes=2, max_height=8, max_width=8, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

## 📚 Module Documentation

### Models
- **`session7_model.py`**: Core model architecture and forward pass
- **`session7_train_model.py`**: Training loop implementation
- **`session7_test_model.py`**: Testing and evaluation functions

### Dataset
- **`session7_dataset.py`**: Data loading, transforms, and CIFAR-10 preparation

### Utils  
- **`session7_utils.py`**: Visualization, analysis, and utility functions

## 🎓 Educational Value

This project demonstrates:
- **Advanced Convolutional Networks**: Modern CNN techniques
- **Data Augmentation**: Albumentations library usage
- **Model Optimization**: Parameter efficiency vs. accuracy trade-offs
- **Best Practices**: Code organization, documentation, and visualization

## 🔄 Migration Guide

### From Old Structure to New Structure

**Old Files (Removed):**
- `models/session7_*` → `models/cifar10_*`, `trainer.py`, `tester.py`
- `dataset/session7_dataset.py` → `data/dataset.py`  
- `utils/session7_utils.py` → `utils/visualization.py`
- `session7_cifar10.ipynb` → `cifar10_training.ipynb`

**New Benefits:**
- ✅ Clean modular structure
- ✅ Command-line training script (`cifar10_trainer.py`)
- ✅ Jupyter notebook for experimentation (`cifar10_training.ipynb`)
- ✅ Centralized configuration (`config.py`)
- ✅ Separate inference script (`infer_cifar10.py`)
- ✅ Enhanced visualization utilities
- ✅ Backward compatibility maintained

### Usage Examples

**Command Line Training:**
```bash
# Quick start
python cifar10_trainer.py

# Extended training  
python cifar10_trainer.py --epochs 50 --batch_size 256 --lr 0.05
```

**Jupyter Notebook:**
```bash
jupyter notebook cifar10_training.ipynb
```

**Model Inference:**
```bash
python infer_cifar10.py --model_path outputs/cifar10_model.pth
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the excellent deep learning framework
- **Albumentations** for powerful data augmentation capabilities
- **CIFAR-10 Dataset** creators for providing this benchmark dataset
- **EVA5** course instructors for guidance on advanced CNN techniques

---

**Project Status**: ✅ All requirements met
**Last Updated**: December 2024
**Author**: Sachin Dangayach
