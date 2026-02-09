import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CatDogDataset(Dataset):
    def __init__(self, config, splits: str):
        self.root_dir = os.path.join(config.tgt_dir, splits)
        self.samples = []
        self.config = config

        self.classes = sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".png")):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
        # chọn sample khác
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.config.height, self.config.width))
        image = image.astype(np.float32) / 255.0

        # HWC → CHW
        image = torch.from_numpy(image).permute(2, 0, 1)

        label = torch.tensor(label, dtype=torch.long)

        return {
            "image": image,
            "label": label
        }
