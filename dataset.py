from torch import FloatTensor, div
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import pickle, torch
import numpy as np

class ImageNetDataset(Dataset):
    """Dataset class for ImageNet"""
    def __init__(self, dataset, labels, transform=None, normalize=None):
        super(ImageNetDataset, self).__init__()
        assert(len(dataset) == len(labels))
        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)

        data = div(data.type(FloatTensor), 255)
        if self.normalize:
            data = self.normalize(data)

        return data, self.labels[idx]

def load_train_data(img_size, magnitude, batch_size):
    with open('train_dataset.pkl', 'rb') as f:
        train_data, train_labels = pickle.load(f)
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandAugment(num_ops=2,magnitude=magnitude),
    ])
    train_dataset = ImageNetDataset(train_data, train_labels.type(torch.LongTensor), transform,
        normalize=transforms.Compose([
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    f.close()
    return train_loader

def load_val_data(img_size, batch_size):
    with open('val_dataset.pkl', 'rb') as f:
        val_data, val_labels = pickle.load(f)
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
    ])
    val_dataset = ImageNetDataset(val_data, val_labels.type(torch.LongTensor), transform,
        normalize=transforms.Compose([
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    f.close()
    return val_loader