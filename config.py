"""
Configuration file for CIFAR-10 CNN training

Centralized configuration management for hyperparameters, 
data settings, and model architecture parameters.
"""

import torch

class Config:
    """Configuration class containing all training and model parameters"""
    
    # Data Configuration
    DATASET_NAME = "CIFAR10"
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 10
    TRAIN_DATA_PATH = "./data"
    
    # Model Configuration
    MODEL_DROPOUT = 0.1
    TARGET_PARAMETERS = 200000  # Parameter limit
    TARGET_ACCURACY = 85.0      # Target accuracy in %
    REJECTIVE_FIELD_TARGET = 44  # Minimum receptive field
    
    # Training Configuration
    DEFAULT_EPOCHS = 20
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_MOMENTUM = 0.9
    DEFAULT_WEIGHT_DECAY = 1e-4
    
    # Learning Rate Scheduler
    LR_SCHEDULER_STEP_SIZE = 10
    LR_SCHEDULER_GAMMA = 0.1
    
    # Device Configuration
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEFAULT_DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    NUM_WORKERS = 4 if CUDA_AVAILABLE else 1
    
    # Data Augmentation (Albumentations)
    AUGMENTATION_PROBABILITIES = {
        "horizontal_flip": 0.5,
        "shift_scale_rotate": 0.5,
        "coarse_dropout": 0.5
    }
    
    AUGMENTATION_PARAMS = {
        "shift_limit": 0.1,
        "scale_limit": 0.1, 
        "rotate_limit": 10,
        "max_holes": 1,
        "min_holes": 1,
        "max_height": 16,
        "max_width": 16,
        "min_height": 16,
        "min_width": 16
    }
    
    # CIFAR-10 Normalization Values (calculated from dataset)
    CIFAR10_MEAN = (0.49, 0.48, 0.45)
    CIFAR10_STD = (0.25, 0.24, 0.26)
    
    # Output Configuration
    DEFAULT_OUTPUT_DIR = "outputs"
    SAVE_PLOTS = True
    PLOT_DPI = 300
    
    # Architecture Configuration
    ARCHITECTURE_BLOCKS = {
        "conv1": {"in_channels": 3, "out_channels": [32, 32], "kernel_size": 3},
        "conv2": {"in_channels": 32, "out_channels": [64, 64], "kernel_size": 3}, 
        "conv3": {"in_channels": 64, "out_channels": [64, 128], "kernel_size": [3, 3], "dilation": [1, 2]},
        "conv4": {"in_channels": 128, "out_channels": [256, 512], "kernel_size": 3, "stride": [1, 2]},
        "gap_fc": {"in_features": 512, "out_features": 10}
    }
    
    # Advanced Convolution Settings
    DEPTHWISE_SEPARABLE_CONV = True
    DILATED_CONV = True
    GLOBAL_AVERAGE_POOLING = True
    NO_MAXPOOLING = True
    
    @classmethod
    def get_device_info(cls):
        """Get device information"""
        if cls.CUDA_AVAILABLE:
            return {
                "device": cls.DEFAULT_DEVICE,
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "num_gpus": torch.cuda.device_count()
            }
        else:
            return {
                "device": cls.DEFAULT_DEVICE,
                "cpu_count": torch.get_num_threads()
            }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 50)
        print("CONFIGURATION SUMMARY")
        print("=" * 50)
        
        device_info = cls.get_device_info()
        print(f"Device: {device_info['device']}")
        
        if cls.CUDA_AVAILABLE:
            print(f"GPU: {device_info['gpu_name']}")
            print(f"CUDA Version: {device_info['cuda_version']}")
        
        print(f"\nTraining Parameters:")
        print(f"- Epochs: {cls.DEFAULT_EPOCHS}")
        print(f"- Batch Size: {cls.DEFAULT_BATCH_SIZE}")
        print(f"- Learning Rate: {cls.DEFAULT_LEARNING_RATE}")
        print(f"- Momentum: {cls.DEFAULT_MOMENTUM}")
        print(f"- Weight Decay: {cls.DEFAULT_WEIGHT_DECAY}")
        
        print(f"\nArchitecture Features:")
        print(f"- Depthwise Separable Conv: {cls.DEPTHWISE_SEPARABLE_CONV}")
        print(f"- Dilated Convolution: {cls.DILATED_CONV}")
        print(f"- Global Average Pooling: {cls.GLOBAL_AVERAGE_POOLING}")
        print(f"- No MaxPooling: {cls.NO_MAXPOOLING}")
        
        print(f"\nTargets:")
        print(f"- Accuracy: {cls.TARGET_ACCURACY}%")
        print(f"- Parameters: <{cls.TARGET_PARAMETERS:,}")
        print(f"- Rejective Field: >{cls.REJECTIVE_FIELD_TARGET}")
        print("=" * 50)


class ExperimentConfig(Config):
    """Configuration for experimental runs with different parameters"""
    
    # Custom experiment settings
    EXPERIMENT_NAME = "cifar10_advanced_convolutions"
    
    # Extended training for better accuracy
    EXTENDED_EPOCHS = 50
    EXTENDED_LR = 0.05
    
    # Smaller model variant
    MINI_MODEL_CONFIG = {
        "conv1": {"in_channels": 3, "out_channels": [16, 16], "kernel_size": 3},
        "conv2": {"in_channels": 16, "out_channels": [32, 32], "kernel_size": 3},
        "conv3": {"in_channels": 32, "out_channels": [32, 64], "kernel_size": [3, 3], "dilation": [1, 2]},
        "conv4": {"in_channels": 64, "out_channels": [128, 256], "kernel_size": 3, "stride": [1, 2]},
        "gap_fc": {"in_features": 256, "out_features": 10}
    }


if __name__ == "__main__":
    config = Config()
    config.print_config()