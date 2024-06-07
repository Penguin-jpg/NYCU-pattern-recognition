import os

import torch
import torchvision.transforms as T
from models import GatedAttentionModel
from pkl_dataset import PickleDataest
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_accuracy, train_val_split, plot_losses, plot_accuracy


def train_one_epoch(model, train_dataloader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0
    total_accuracy = 0
    for images, label in tqdm(train_dataloader):
        images = images.to(device)
        label = label.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        predicted = model(images)
        loss = loss_fn(predicted, label)
        loss.backward()
        accuracy = get_accuracy(predicted, label)
        total_loss += loss.item()
        total_accuracy += accuracy
        optimizer.step()

    total_loss /= len(train_dataloader)
    total_accuracy /= len(train_dataloader)

    return total_loss, total_accuracy


@torch.inference_mode()
def evaluate(model, val_dataloader, loss_fn, device):
    model.eval()

    total_loss = 0
    total_accuracy = 0
    for images, label in tqdm(val_dataloader):
        images = images.to(device)
        label = label.to(device, dtype=torch.float32)
        predicted = model(images)
        loss = loss_fn(predicted, label)
        accuracy = get_accuracy(predicted, label)
        total_loss += loss.item()
        total_accuracy += accuracy

    total_loss /= len(val_dataloader)
    total_accuracy /= len(val_dataloader)

    return total_loss, total_accuracy


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    num_epochs,
    checkpoint_path,
    device,
):
    train_losses, val_losses = [], []
    training_accuracy, valid_accuracy = [], []
    best_accuracy = None
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model, train_dataloader, optimizer, loss_fn, device
        )
        train_losses.append(train_loss)
        training_accuracy.append(train_accuracy.item())
        val_loss, val_accuracy = evaluate(model, val_dataloader, loss_fn, device)
        val_losses.append(val_loss)
        valid_accuracy.append(val_accuracy.item())
        print(
            f"Epoch {epoch} Train loss: {train_loss:.4f} Train accuracy: {train_accuracy:.4f} Val loss: {val_loss:.4f} Val accuracy: {val_accuracy:.4f}"
        )

        if best_accuracy is None or val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                model.state_dict(), os.path.join(checkpoint_path, "best_model.pth")
            )
            print(f"Accuracy {best_accuracy}. Saving model to best_model.pth")

        torch.save(model.state_dict(), os.path.join(checkpoint_path, f"{epoch}.pth"))

    plot_losses(train_losses, val_losses)
    plot_accuracy(training_accuracy, valid_accuracy)


if __name__ == "__main__":
    num_epochs = 10
    batch_size = 1
    checkpoint_path = "checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_bags, val_bags = train_val_split(
        image_dir=os.path.join("dataset", "train"), ratio=0.8
    )
    train_dataset = PickleDataest(train_bags, transform)
    val_dataset = PickleDataest(val_bags, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedAttentionModel(instance_dim=512, hidden_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_fn,
        num_epochs,
        checkpoint_path,
        device,
    )
