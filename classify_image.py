import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn
from PIL import Image
import requests
from io import BytesIO


def get_model(model_name):
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(torchvision.datasets.ImageFolder("dataset/train").classes)),
        nn.LogSoftmax(dim=1))

    model.load_state_dict(torch.load(model_name))
    return model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Used device:", device.type)

url = "https://thumbs.img-sprzedajemy.pl/1000x901c/67/7e/61/szczeniaki-dalmatynczyk-pozostale-swietokrzyskie-556906592.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
print("Image loaded")

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(p=0.5)
])
trainset = torchvision.datasets.ImageFolder("dataset/train", transform=transforms)
model = get_model('./model1.pt')
model.to(device)

print("Model loaded")
model.eval()


def predictor(img, n=5):
    # transform the image
    img = transforms(img)
    # get the class predicted
    pred = int(np.squeeze(model(img.unsqueeze(0).to(device)).data.max(1, keepdim=True)[1].cpu().numpy()))
    # the number is also the index for the class label
    pred = trainset.classes[pred]
    # get model log probabilities
    preds = torch.from_numpy(np.squeeze(model(img.unsqueeze(0).to(device)).data.cpu().numpy()))
    # convert to prediction probabilities of the top n predictions
    top_preds = torch.topk(torch.exp(preds), n)
    # display at an orgenized fasion
    top_preds = dict(zip([trainset.classes[i] for i in top_preds.indices],
                         [f"{round(float(i) * 100, 2)}%" for i in top_preds.values]))
    return pred, top_preds


print("-------------------------")
print("Prediction: ")
my_prediction, top_predictions = predictor(img, n=5)

print(my_prediction)
print(top_predictions)
