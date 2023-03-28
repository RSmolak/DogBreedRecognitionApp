import torchvision
import torchvision.models as models
import torch
import torch.nn as nn


def get_model(model_name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(torchvision.datasets.ImageFolder("dataset/train").classes)),
        nn.LogSoftmax(dim=1))

    model.to(device)
    model.load_state_dict(torch.load(model_name, map_location=device))
    return model
