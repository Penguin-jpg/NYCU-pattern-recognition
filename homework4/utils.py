import os
import pickle
import random
from glob import glob

import torch
import matplotlib.pyplot as plt


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


def get_accuracy(predicted, label):
    predicted = torch.round(predicted)
    correct = (predicted == label).sum().float()
    accuracy = correct / len(label)
    return accuracy


def train_val_split(image_dir, ratio=0.8):
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

    return train_bags, val_bags


def get_test_bags(image_dir):
    # since test dataset doesn't have labels, we use -1 instead
    return [Bag(pkl_path, -1) for pkl_path in glob(os.path.join(image_dir, "*.pkl"))]


def plot_losses(train_losses, val_losses):
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("losses.png")
    plt.show()
    plt.clf()


def plot_accuracy(train_accuracy, val_accuracy):
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(train_accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.legend()
    plt.savefig("accuracy.png")
    plt.show()
    plt.clf()
