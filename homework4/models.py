# code based on https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GatedAttentionModel(nn.Module):
    def __init__(self, instance_dim, hidden_dim):
        super(GatedAttentionModel, self).__init__()
        # use pre-trained ResNet-18 as feature extractor
        self.cnn = models.resnet18(pretrained=True)
        # we don't need the last fc layer so change it to identity
        self.cnn.fc = nn.Identity()

        self.instance_encoder = nn.Sequential(
            nn.Linear(instance_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # gated attention mechanism
        self.attention_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, bag):
        B, N, C, H, W = bag.shape
        # flatten the batch dimension
        bag = bag.view(-1, C, H, W)

        # extract features and reshape to [B, N, instance_dim]
        instance_features = self.cnn(bag).view(B, N, -1)
        instance_features = instance_features.view(B, N, -1)

        # a small MLP to reduce the dimension
        instance_embeddings = self.instance_encoder(instance_features)

        # calculate attention
        attention_v = self.attention_v(instance_embeddings)
        attention_u = self.attention_u(instance_embeddings)
        # element-wise multiplication for gating
        attention_weights = self.attention_weights(attention_v * attention_u)
        # normalize weights along the instance dimension
        attention_weights = F.softmax(attention_weights, dim=1)

        # aggregate instances
        bag_embedding = torch.sum(attention_weights * instance_embeddings, dim=1)

        # classify
        out = self.classifier(bag_embedding).sigmoid()
        return out


class GatedAttentionModel2(nn.Module):
    def __init__(self, hidden_dim):
        super(GatedAttentionModel2, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        # gated attention mechanism
        self.attention_v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, bag):
        B, N, C, H, W = bag.shape
        # flatten the batch dimension
        bag = bag.view(-1, C, H, W)

        # extract features and reshape to [B, N, instance_dim]
        instance_features = self.cnn(bag).view(B, N, -1)
        instance_features = instance_features.view(B, N, -1)

        # calculate attention
        attention_v = self.attention_v(instance_features)
        attention_u = self.attention_u(instance_features)
        # element-wise multiplication for gating
        attention_weights = self.attention_weights(attention_v * attention_u)
        # normalize weights along the instance dimension
        attention_weights = F.softmax(attention_weights, dim=1)

        # aggregate instances
        bag_embedding = torch.sum(attention_weights * instance_features, dim=1)

        # classify
        out = self.classifier(bag_embedding).sigmoid()
        return out


class GatedAttentionModel3(nn.Module):
    def __init__(self, instance_dim, hidden_dim):
        super(GatedAttentionModel3, self).__init__()
        # use pre-trained ResNet-18 as feature extractor
        self.cnn = models.resnet18(pretrained=True)
        # we don't need the last fc layer so change it to identity
        self.cnn.fc = nn.Identity()

        self.instance_encoder = nn.Sequential(
            nn.Linear(instance_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(hidden_dim * hidden_dim, 1)

    def forward(self, bag):
        B, N, C, H, W = bag.shape
        # flatten the batch dimension
        bag = bag.view(-1, C, H, W)

        # extract features and reshape to [B, N, instance_dim]
        instance_features = self.cnn(bag).view(B, N, -1)
        instance_features = instance_features.view(B, N, -1)

        # a small MLP to reduce the dimension
        instance_embeddings = self.instance_encoder(instance_features)
        instance_embeddings = instance_embeddings.view(B, -1)

        # classify
        out = self.classifier(instance_embeddings).sigmoid()
        return out
