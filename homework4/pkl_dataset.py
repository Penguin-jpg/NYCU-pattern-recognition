import pickle
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
import torch
import torchvision.transforms as T
import numpy as np


class Bag:
    def __init__(self, pkl_path, label):
        self.pkl_path = pkl_path
        self.label = label


class PickleDataest(Dataset):
    def __init__(self, image_dir, transform=None, split="train"):
        super(PickleDataest, self).__init__()

        self.bags = []
        for pkl_path in glob(os.path.join(image_dir, "train", "class_0", "*.pkl")):
            self.bags.append(Bag(pkl_path, 0))
        for pkl_path in glob(os.path.join(image_dir, "train", "class_1", "*.pkl")):
            self.bags.append(Bag(pkl_path, 1))

        if split == "train":
            self.bags = self.bags[: int(len(self.bags) * 0.8)]
        else:
            self.bags = self.bags[int(len(self.bags) * 0.8) :]

        self.transform = transform

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]

        # each bag contains 256 images with shape (128, 128, 3)
        # shape: [256, 128, 128, 3]
        with open(bag.pkl_path, "rb") as file:
            data = pickle.load(file)

        images = []
        if self.transform is not None:
            # data = torch.from_numpy(data).permute(0, 3, 1, 2).to(dtype=torch.float32)
            for i in range(data.shape[0]):
                images.append(self.transform(data[i]))
        else:
            to_tensor = T.ToTensor()
            for i in range(data.shape[0]):
                images.append(to_tensor(data[i]))

        images = torch.stack(images)
        label = torch.as_tensor([bag.label])

        return images, label


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = PickleDataest(image_dir="dataset", transform=transform, split="train")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(len(dataset))

    for bag, label in dataloader:
        print(bag)
        print(label)
