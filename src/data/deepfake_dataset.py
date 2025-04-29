import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        """
        Args:
            real_dir (str): Directory with real images
            fake_dir (str): Directory with fake images
            transform: Optional transform to be applied on images
        """
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((299, 299)),  # Standard size for Xception/InceptionV3
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Get list of all images
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir) if img.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Combine all images and create labels
        self.image_paths = self.real_images + self.fake_images
        self.labels = ([1] * len(self.real_images)) + ([0] * len(self.fake_images))  # 1 for real, 0 for fake
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
