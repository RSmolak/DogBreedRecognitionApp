import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import torchvision
import matplotlib.pyplot as plt
import torchvision.models as models
import tqdm as tq
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO


# Hyper parameters
in_chan = 3
num_class = 70
num_epochs = 100
batch_size = 64
learning_rate = 0.001

# Choosing learning device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Used device:", device.type)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomHorizontalFlip(p=0.5)
])
# datasets
train_set = torchvision.datasets.ImageFolder("dataset/train", transform=transforms)
valid_set = torchvision.datasets.ImageFolder("dataset/valid", transform=transforms)

# loaders for data
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)

images, labels = iter(train_loader).__next__()
images, labels = images.numpy(), labels.numpy()

fig = plt.figure(figsize=(15, 5))

for i in range(int(batch_size/8)):
    ax = fig.add_subplot(2, int(batch_size/16), i + 1, xticks=[], yticks=[])
    ax.imshow(np.transpose(images[i], (1, 2, 0)), cmap='gray')
    ax.set_title(train_set.classes[labels[i]] + " ")

# plt.show()

# Initialize network
# model = CNN().to(device)
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features

for param in model.parameters():
    param = param.requires_grad_(False)

model.fc = nn.Sequential(nn.Dropout(0.5),
                      nn.Linear(model.fc.in_features, 256),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(256, len(train_set.classes)),
                      nn.LogSoftmax(dim=1))

model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)




def training_loop(model,
                  criterion,
                  optimizer,
                  train_loader,
                  valid_loader,
                  trained_model_path,
                  num_epochs,
                  device,
                  max_no_improve_epochs):

    valid_loss_min = np.Inf
    no_improve_epochs = 0
    best_valid_loss = np.Inf
    best_valid_acc = 0

    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)
            print('\r', end='')
            print(f'Epoch: {epoch + 1}\t{100 * (batch_idx + 1) / len(train_loader):.2f}% complete.', end='')
        else:
            # model.epochs += 1
            with torch.no_grad():
                model.eval()
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target.long())
                    valid_loss += loss.item() * data.size(0)

                    # Calculate accuracy of validation set
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                if valid_loss < valid_loss_min:
                    valid_loss_min = valid_loss
                    torch.save(model.state_dict(), trained_model_path)
                    no_improve_epochs = 0
                    best_valid_loss = valid_loss
                    best_valid_acc = valid_acc / len(valid_loader.dataset)
                else:
                    no_improve_epochs += 1

                print(
                    f'\n\nEpoch: {epoch + 1}\tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%\n'
                )

                if no_improve_epochs == max_no_improve_epochs:
                    print(f"Learning stopped due to no improvement in learning for {max_no_improve_epochs} epochs!")
                    print(f"Trained model has:")
                    print(f"Validation Loss: {best_valid_loss:.4f}")
                    print(f"Validation Accuracy: {100 * best_valid_acc:.2f}%")
                    break
    return model


model = training_loop(
    model,
    criterion,
    optimizer,
    train_loader,
    valid_loader,
    trained_model_path="./model1.pt",
    num_epochs=num_epochs,
    device=device,
    max_no_improve_epochs=5)
