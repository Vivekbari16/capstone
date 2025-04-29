"""
Enhanced data loader for DeepFake detection with advanced augmentation techniques
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class DeepfakeDatasetEnhanced(Dataset):
    """
    Enhanced dataset class for DeepFake detection with advanced augmentation
    """
    def __init__(self, real_dir, fake_dir, transform=None, img_size=224, apply_mixup=False, mixup_alpha=0.2):
        """
        Args:
            real_dir (str): Directory with real images
            fake_dir (str): Directory with fake images
            transform: Albumentations transform pipeline
            img_size (int): Target image size
            apply_mixup (bool): Whether to apply mixup augmentation
            mixup_alpha (float): Alpha parameter for mixup
        """
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.img_size = img_size
        self.apply_mixup = apply_mixup
        self.mixup_alpha = mixup_alpha
        
        # Default transform if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
        
        # Get list of all images
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) 
                           if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) 
                           if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Combine all images and create labels
        self.image_paths = self.real_images + self.fake_images
        self.labels = ([1] * len(self.real_images)) + ([0] * len(self.fake_images))  # 1 for real, 0 for fake
        
        # Print dataset statistics
        print(f"Dataset loaded with {len(self.real_images)} real images and {len(self.fake_images)} fake images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # Apply mixup if enabled (during training)
        if self.apply_mixup and torch.rand(1) < 0.5:
            # Get another random sample
            mix_idx = torch.randint(0, len(self), (1,)).item()
            mix_img_path = self.image_paths[mix_idx]
            mix_image = np.array(Image.open(mix_img_path).convert('RGB'))
            mix_label = self.labels[mix_idx]
            
            if self.transform:
                mix_transformed = self.transform(image=mix_image)
                mix_image = mix_transformed["image"]
            
            # Generate mixup coefficient
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            # Apply mixup
            image = lam * image + (1 - lam) * mix_image
            return image, torch.tensor([label, mix_label, lam], dtype=torch.float)
        
        return image, torch.tensor(label)

def get_train_transforms(img_size=224):
    """
    Get training transforms with advanced augmentation
    """
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.8, 1.0), size=img_size),
        A.Rotate(limit=30),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.2),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50)),
            A.ISONoise(),
        ], p=0.2),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.HueSaturationValue(),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(img_size=224):
    """
    Get validation transforms
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_test_transforms(img_size=224):
    """
    Get test transforms
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def mixup_criterion(criterion, pred, targets):
    """
    Mixup loss function
    """
    targets1, targets2, lam = targets[:, 0], targets[:, 1], targets[:, 2]
    return lam * criterion(pred, targets1.long()) + (1 - lam) * criterion(pred, targets2.long())

def create_class_weights(labels):
    """
    Create class weights to handle class imbalance
    """
    class_counts = torch.bincount(torch.tensor(labels))
    total_samples = sum(class_counts)
    weights = total_samples / (len(class_counts) * class_counts)
    return weights

def get_data_loaders(real_dir, fake_dir, batch_size=16, img_size=224, train_ratio=0.7, 
                     val_ratio=0.15, num_workers=4, apply_mixup=True, mixup_alpha=0.2):
    """
    Create train, validation, and test data loaders with advanced augmentation
    
    Args:
        real_dir (str): Directory with real images
        fake_dir (str): Directory with fake images
        batch_size (int): Batch size
        img_size (int): Target image size
        train_ratio (float): Ratio of data for training
        val_ratio (float): Ratio of data for validation
        num_workers (int): Number of workers for data loading
        apply_mixup (bool): Whether to apply mixup augmentation
        mixup_alpha (float): Alpha parameter for mixup
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Get all image paths and labels
    real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) 
                  if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) 
                  if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    image_paths = real_images + fake_images
    labels = ([1] * len(real_images)) + ([0] * len(fake_images))  # 1 for real, 0 for fake
    
    # Split data into train, validation, and test sets
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=(1-train_ratio-val_ratio), random_state=42, stratify=labels
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_ratio/(train_ratio+val_ratio), 
        random_state=42, stratify=train_val_labels
    )
    
    # Create datasets
    train_dataset = CustomDataset(
        train_paths, train_labels, transform=get_train_transforms(img_size), 
        apply_mixup=apply_mixup, mixup_alpha=mixup_alpha
    )
    
    val_dataset = CustomDataset(
        val_paths, val_labels, transform=get_val_transforms(img_size)
    )
    
    test_dataset = CustomDataset(
        test_paths, test_labels, transform=get_test_transforms(img_size)
    )
    
    # Calculate class weights for weighted loss
    class_weights = create_class_weights(train_labels)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Data loaders created with:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Class weights: {class_weights}")
    
    return train_loader, val_loader, test_loader, class_weights

class CustomDataset(Dataset):
    """
    Custom dataset for loading images from paths and labels
    """
    def __init__(self, image_paths, labels, transform=None, apply_mixup=False, mixup_alpha=0.2):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.apply_mixup = apply_mixup
        self.mixup_alpha = mixup_alpha
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = self.labels[idx]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # Apply mixup if enabled (during training)
        if self.apply_mixup and torch.rand(1) < 0.5:
            # Get another random sample
            mix_idx = torch.randint(0, len(self), (1,)).item()
            mix_img_path = self.image_paths[mix_idx]
            mix_image = np.array(Image.open(mix_img_path).convert('RGB'))
            mix_label = self.labels[mix_idx]
            
            if self.transform:
                mix_transformed = self.transform(image=mix_image)
                mix_image = mix_transformed["image"]
            
            # Generate mixup coefficient
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            # Apply mixup
            image = lam * image + (1 - lam) * mix_image
            return image, torch.tensor([label, mix_label, lam], dtype=torch.float)
        
        return image, torch.tensor(label)
