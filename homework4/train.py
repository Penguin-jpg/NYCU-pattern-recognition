import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from model import Attention, GatedAttention
from pkl_dataset import PickleDataest


def train_one_epoch(model, train_dataloader, optimizer, device):
    model.train()

    total_loss = 0
    total_error = 0
    for bag, label in train_dataloader:
        bag = bag.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        loss, _ = model.calculate_objective(bag, label)
        loss.backward()
        error, _ = model.calculate_classification_error(bag, label)
        total_loss += loss.item()
        total_error += error
        optimizer.step()

    total_loss /= len(train_dataloader)
    total_error /= len(train_dataloader)

    return total_loss, total_loss


@torch.inference_mode()
def evaluate(model, val_dataloader, device):
    model.eval()

    total_loss = 0
    total_error = 0
    for bag, label in val_dataloader:
        bag = bag.to(device)
        label = label.to(device)

        loss, _ = model.calculate_objective(bag, label)
        error, _ = model.calculate_classification_error(bag, label)
        total_loss += loss.item()
        total_error += error

    total_loss /= len(val_dataloader)
    total_error /= len(val_dataloader)

    return total_loss, total_loss


def train(model, train_dataloader, val_dataloader, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        train_loss, train_error = train_one_epoch(
            model, train_dataloader, optimizer, device
        )
        val_loss, val_error = evaluate(model, val_dataloader, device)
        print(
            f"Epoch {epoch} Train loss: {train_loss:.4f} Train error: {train_error:.4f} Val loss: {val_loss:.4f} Val error: {val_error:.4f}"
        )


if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = PickleDataest(
        image_dir="dataset", transform=transform, split="train"
    )
    val_dataset = PickleDataest(image_dir="dataset", transform=transform, split="val")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Attention()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train(model, train_dataloader, val_dataloader, optimizer, 10, device)
