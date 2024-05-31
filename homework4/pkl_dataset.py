import pickle
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import torch
import torchvision.transforms as T
import random


class Bag:
    def __init__(self, pkl_path, label):
        self.pkl_path = pkl_path
        self.label = label

    def open_bag(self):
        # each bag contains 256 images with shape (128, 128, 3)
        # shape: [256, 128, 128, 3]
        with open(self.pkl_path, "rb") as file:
            data = pickle.load(file)

        return data


class PickleDataest(Dataset):
    def __init__(self, bags, transform=None, return_image_id=False):
        super(PickleDataest, self).__init__()

        self.bags = bags
        self.transform = transform
        self.return_image_id = return_image_id

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        data = bag.open_bag()

        images = []
        if self.transform is not None:
            for i in range(data.shape[0]):
                images.append(self.transform(data[i]))
        else:
            to_tensor = T.ToTensor()
            for i in range(data.shape[0]):
                images.append(to_tensor(data[i]))

        images = torch.stack(images)
        label = torch.as_tensor([bag.label])

        if not self.return_image_id:
            return images, label
        else:
            return images, label, [os.path.splitext(os.path.basename(bag.pkl_path))[0]]


def train_val_split(image_dir, ratio=0.8, transform=None):
    class_0_bags = [
        Bag(pkl_path, 0)
        for pkl_path in glob(os.path.join(image_dir, "class_0", "*.pkl"))
    ]
    class_1_bags = [
        Bag(pkl_path, 1)
        for pkl_path in glob(os.path.join(image_dir, "class_1", "*.pkl"))
    ]
    bags = class_0_bags + class_1_bags
    random.shuffle(bags)

    train_bags = bags[: int(len(bags) * ratio)]
    val_bags = bags[int(len(bags) * ratio) :]

    train_dataset = PickleDataest(train_bags, transform)
    val_dataset = PickleDataest(val_bags, transform)

    return train_dataset, val_dataset


def get_test_dataset(image_dir, transform=None):
    # since test dataset doesn't have labels, we use -1 instead
    test_bags = [
        Bag(pkl_path, -1) for pkl_path in glob(os.path.join(image_dir, "*.pkl"))
    ]
    test_dataset = PickleDataest(test_bags, transform, return_image_id=True)
    return test_dataset


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset, val_dataset = train_val_split(
        os.path.join("dataset", "train"), ratio=0.8, transform=transform
    )
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    print(len(train_dataset))
    print(len(val_dataset))

    for bag, label in train_dataloader:
        print(bag)
        print(label)
