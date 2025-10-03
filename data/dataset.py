# CIFAR-10 Dataset with Albumentations
import torch
import torchvision
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CIFAR10Albumentation:
    """Custom CIFAR-10 dataset wrapper for Albumentations compatibility"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image=image)['image']
            
        return image, label


class CIFAR10DataLoader:
    """CIFAR-10 data loader with Albumentations support"""
    
    def __init__(self, data_path='./data', batch_size=128, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_cifar10_stats(self):
        """Calculate CIFAR-10 dataset statistics"""
        print("Calculating CIFAR-10 normalization statistics...")
        
        # Load raw data
        simple_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        
        train_data = torchvision.datasets.CIFAR10(
            self.data_path, train=True, download=True, transform=simple_transforms
        )
        test_data = torchvision.datasets.CIFAR10(
            self.data_path, train=False, download=True, transform=simple_transforms
        )
        
        # Combine datasets
        all_data = torch.cat([train_data.data, test_data.data])
        all_data = np.array(all_data).astype(np.float32) / 255.0
        
        # Calculate mean and std
        mean = np.mean(all_data, axis=(0, 1, 2))
        std = np.std(all_data, axis=(0, 1, 2))
        
        return tuple(mean), tuple(std)
    
    def get_transforms(self, norm_mean, norm_std):
        """Get Albumentations transforms"""
        print(f"Creating transforms with normalization: mean={norm_mean}, std={norm_std}")
        
        # Training transforms with augmentations
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=list(norm_mean),
                mask_fill_value=None,
                p=0.5
            ),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ])
        
        # Test transforms (no augmentation)
        test_transform = A.Compose([
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ])
        
        return train_transform, test_transform
    
    def get_datasets(self, train_transform, test_transform):
        """Get train and test datasets"""
        # Load raw datasets
        raw_train = torchvision.datasets.CIFAR10(root=self.data_path, train=True, download=True)
        raw_test = torchvision.datasets.CIFAR10(root=self.data_path, train=False, download=True)
        
        # Convert to numpy for Albumentations
        train_images = np.array(raw_train.data)
        train_labels = raw_train.targets
        
        test_images = np.array(raw_test.data)
        test_labels = raw_test.targets
        
        # Create custom datasets
        train_dataset = CIFAR10Albumentation(
            image=train_images, label=train_labels, transform=train_transform
        )
        test_dataset = CIFAR10Albumentation(
            image=test_images, label=test_labels, transform=test_transform
        )
        
        return train_dataset, test_dataset, raw_train.classes
    
    def get_data(self):
        """Main method to get complete data pipeline"""
        # Get normalization statistics
        norm_mean, norm_std = self.get_cifar10_stats()
        
        # Get transforms
        train_transform, test_transform = self.get_transforms(norm_mean, norm_std)
        
        # Get datasets
        train_dataset, test_dataset, classes = self.get_datasets(train_transform, test_transform)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        norm_params = {
            'mean': norm_mean,
            'std': norm_std
        }
        
        return train_loader, test_loader, classes, norm_params


# Legacy functions for backward compatibility
def cifar10_mean_std():
    """Legacy function"""
    loader = CIFAR10DataLoader()
    return loader.get_cifar10_stats()


def get_transforms(norm_mean, norm_std):
    """Legacy function"""
    loader = CIFAR10DataLoader()
    return loader.get_transforms(norm_mean, norm_std)


def get_datasets(train_transform, test_transform):
    """Legacy function"""
    loader = CIFAR10DataLoader()
    return loader.get_datasets(train_transform, test_transform)


def get_dataloaders(train_set, test_set):
    """Legacy function"""
    SEED = 1
    torch.manual_seed(SEED)
    
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(SEED)
    
    dataloader_args = dict(
        shuffle=True, 
        batch_size=128, 
        num_workers=4, 
        pin_memory=True
    ) if cuda else dict(shuffle=True, batch_size=64, num_workers=1)
    
    train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)
    
    return train_loader, test_loader
