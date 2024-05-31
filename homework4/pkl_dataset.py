import os

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from utils import train_val_split


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


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_bags, val_bags = train_val_split(os.path.join("dataset", "train"), ratio=0.8)
    train_dataset = PickleDataest(train_bags, transform)
    val_dataset = PickleDataest(val_bags, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    print(len(train_dataset))
    print(len(val_dataset))

    for bag, label in train_dataloader:
        print(bag)
        print(label)
