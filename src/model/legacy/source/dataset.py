import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

TARGET_SIZE = (512, 512)

class DamageDataset(Dataset):
    def __init__(self, root):
        self.pre_dir = os.path.join(root, "pre")
        self.post_dir = os.path.join(root, "post")
        self.mask_dir = os.path.join(root, "mask")

        self.ids = sorted([
            f.replace("pre.png", "")
            for f in os.listdir(self.pre_dir)
            if f.endswith("pre.png")
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        idx_id = self.ids[idx]

        pre = cv2.imread(os.path.join(self.pre_dir, f"{idx_id}pre.png"), cv2.IMREAD_GRAYSCALE)
        post = cv2.imread(os.path.join(self.post_dir, f"{idx_id}post.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir, f"{idx_id}mask.png"), cv2.IMREAD_GRAYSCALE)

        pre = cv2.resize(pre, TARGET_SIZE)
        post = cv2.resize(post, TARGET_SIZE)
        mask = cv2.resize(mask, TARGET_SIZE)

        pre = pre / 255.0
        post = post / 255.0
        mask = (mask > 0).astype("float32")

        x = np.stack([pre, post], axis=0)
        x = torch.from_numpy(x).float()
        y = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return x, y
