from typing import Optional
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from common.modules.classifier import Classifier as ClassifierBase
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.sequence = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.sequence(x)
        return x


class CosineSim(nn.Module):
    def __init__(self, k, rou):
        super(CosineSim, self).__init__()
        self.cosine = nn.CosineSimilarity(dim=2)
        self.softmax = nn.Softmax(dim=1)
        self.rou = rou
        self.k = k

    def forward(self, x, centroids):
        x = x.unsqueeze(1).repeat(1, self.k, 1)
        x = self.cosine(x, centroids)
        x = self.softmax(self.rou * x)
        return x


class ClusterDistribution:

    def __init__(self, target_loader, backbone, k=25, rou=1):

        self.backbone = backbone
        self.centroids = self.cluster(target_loader, k)  # calculate cluster centroid
        self.c = deepcopy(self.centroids)
        self.cosine_sim = CosineSim(k, rou).to(device)  # calculate cluster distribution
        self.num_features = self.centroids.shape[1]

    def cluster(self, target_loader, k=25):
        features = []
        with torch.no_grad():
            for data in target_loader:
                features.append(self.backbone(data[0]))
        features = torch.vstack(features)
        k_means = KMeans(n_clusters=k, random_state=0).fit(features)
        centroids = torch.from_numpy(k_means.cluster_centers_)
        return centroids

    def calculate(self, x):
        with torch.no_grad():
            x = self.backbone(x)
            if self.c.shape[0] != x.shape[0]:
                self.c = self.centroids.repeat(x.shape[0], 1, 1)
            x = self.cosine_sim(x, self.c)
        return x



class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 512, **kwargs):
        bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            # nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)


class ASoftmax(nn.Module):
    def __init__(self, backbone, num_clusters: int, num_features: int):
        super(ASoftmax, self).__init__()
        self.W = nn.Parameter(torch.randn(num_clusters, num_features))
        self.Wmatrix = deepcopy(self.W)
        self.backbone = backbone
        self.cosine_sim = CosineSim(num_clusters, 1).to(device)

    def forward(self, x):
        x = self.backbone(x)
        if self.W.shape[0] != x.shape[0]:
            self.W = self.Wmatrix.repeat(x.shape[0], 1, 1)
        x = self.cosine_sim(x, self.W)
        return x

    def get_parameters(self):
        params = [
            {"params": self.W, "lr": 1.0}
        ]
        return params

class ConditionalEntropyLoss(nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        x = -torch.sum(x * torch.log(x))
        return x


# if __name__ == '__main__':
#     conditional_entropy = ConditionalEntropyLoss()
#     x = torch.tensor(abs(torch.randn(11,1)))
#     print(x)
#     loss = conditional_entropy(x)
#     print(loss)
#     c = torch.tensor([[1,1],[2,10],[3,3]])
#     d = torch.tensor([[1.1,1.1],[4,2.2]])
#
#     cos = CosineSim(3,1)
#
#     if c.shape[0] != d.shape[0]:
#         c = c.repeat(d.shape[0],1,1)
#
#     a = cos(d, c)
