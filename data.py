import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class DiffSet(Dataset):
    def __init__(self, is_train, dataset_path):
        self.is_train = is_train
        self.dataset_path = dataset_path
        self.size = 32  # Set this to the desired image size
        self.depth = 3  # Set this to the number of image channels (e.g., 3 for RGB)
        self.image_paths = self.load_image_paths()

    def load_image_paths(self):
        # Modify this to load images from your dataset directory
        image_paths = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(("png", "jpg", "jpeg")):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.size, self.size))  # Resize to the desired size
        img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.tensor(img).permute(2, 0, 1)  # Convert to (C, H, W) format
        return img