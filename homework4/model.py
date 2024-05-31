# code based on https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MilModel(nn.Module):
    def __init__(self):
        super(MilModel, self).__init__()

        extractor_weight = models.ResNet50_Weights.DEFAULT
        self.feature_extractor = models.resnet50(weights=extractor_weight)
        # remove the last fc layer
        self.feature_extractor = nn.Sequential(
            *list(self.feature_extractor.children())[:-1]
        )
        self.feature_extractor.eval()
        self.preprocess = extractor_weight.transforms()

        self.bottleneck = nn.Linear(2048, 256)
        self.out_mlp = nn.Sequential(
            nn.Linear(256 * 256, 128 * 128),
            nn.Linear(128 * 128, 32 * 32),
            nn.Linear(32 * 32, 1),
        )

    def forward(self, x):
        # x = self.preprocess(x)
        x = x.view(*x.shape[1:])
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.bottleneck(x)
        x = x.view(-1)
        # print(x.shape)
        x = self.out_mlp(x)
        # print(x.shape)
        return x.sigmoid()


# class GatedAttentionModel(nn.Module):
#     def __init__(
#         self,
#         encoder_channels=[64, 128, 256],
#         dim=512,
#         attention_dim=128,
#         num_attention_branches=1,
#     ):
#         super(GatedAttentionModel, self).__init__()

#         self.instance_encoder = nn.Sequential(
#             nn.Conv2d(
#                 3,
#                 encoder_channels[0],
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(
#                 encoder_channels[0],
#                 encoder_channels[1],
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(
#                 encoder_channels[1],
#                 encoder_channels[2],
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=2),
#         )

#         self.bottleneck = nn.Sequential(
#             nn.Linear(encoder_channels[2] * 16 * 16, dim),
#             nn.ReLU(),
#         )

#         self.attention_v = nn.Sequential(
#             nn.Linear(dim, attention_dim),
#             nn.Tanh(),
#         )

#         self.attention_u = nn.Sequential(
#             nn.Linear(dim, attention_dim),
#             nn.Sigmoid(),
#         )

#         self.attention_weights = nn.Linear(attention_dim, num_attention_branches)
#         self.classify = nn.Sequential(
#             nn.Linear(dim * num_attention_branches, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         # remove the batch dimension
#         x = x.squeeze(0)

#         h = self.instance_encoder(x)
#         h = h.view(-1, 256 * 16 * 16)
#         h = self.bottleneck(h)

#         v = self.attention_v(h)
#         u = self.attention_u(h)
#         # element-wise multiplication
#         attention = self.attention_weights(v * u).transpose(1, 0)
#         attention = F.softmax(attention, dim=1)
#         attention_out = torch.matmul(attention, h)

#         out = self.classify(attention_out)
#         return out


class GatedAttentionModel(nn.Module):
    def __init__(self, instance_dim, hidden_dim):
        super(GatedAttentionModel, self).__init__()
        # use pre-trained ResNet-18
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the classification layer
        self.instance_encoder = nn.Sequential(
            nn.Linear(instance_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.attention_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, bag):
        B, N, C, H, W = bag.shape
        bag = bag.view(-1, C, H, W)  # Flatten the bag for CNN processing
        instance_features = self.cnn(bag)  # Extract features with CNN
        instance_features = instance_features.view(
            B, N, -1
        )  # Reshape to (batch_size, num_instances, instance_dim)
        instance_embeddings = self.instance_encoder(
            instance_features
        )  # Encode instances
        attention_v = self.attention_v(instance_embeddings)
        attention_u = self.attention_u(instance_embeddings)
        attention_weights = self.attention_weights(
            attention_v * attention_u
        )  # Element-wise multiplication for gating
        attention_weights = F.softmax(
            attention_weights, dim=1
        )  # Normalize weights along the instance dimension
        bag_embedding = torch.sum(
            attention_weights * instance_embeddings, dim=1
        )  # Aggregate instances
        bag_output = self.classifier(bag_embedding)  # Classify bag
        bag_output = torch.sigmoid(bag_output)
        return bag_output
