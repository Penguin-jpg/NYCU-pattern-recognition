import torch
from model import GatedAttentionModel
import torchvision.transforms as T
from utils import get_test_bags
from pkl_dataset import PickleDataest
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

if __name__ == "__main__":
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_bags = get_test_bags(os.path.join("dataset", "test"))
    test_dataset = PickleDataest(test_bags, transform, return_image_id=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GatedAttentionModel(instance_dim=512, hidden_dim=256)
    model.load_state_dict(torch.load(os.path.join("checkpoints", "best_model.pth")))
    model.to(device)
    model.eval()

    table = [["image_id", "y_pred"]]

    with torch.no_grad():
        for images, _, image_id in tqdm(test_dataloader):
            images = images.to(device)
            predicted = model(images)
            table.append([image_id[0][0], torch.round(predicted).item()])

    with open("submission.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerows(table)
