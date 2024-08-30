import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

# Data Preprocessing
transform = transforms.Compose([
    # Convert grayscale to 3-channel image
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Loading Data
train_data = datasets.ImageFolder(
    './Dataset/Augmented Images/Augmented Images', transform=transform)
valid_data = datasets.ImageFolder(
    './Dataset/Original Images', transform=transform)

# Splitting validation into validation and test sets
num_valid = len(valid_data)
indices = list(range(num_valid))
split = int(0.4 * num_valid)

valid_idx, test_idx = indices[split:], indices[:split]
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, sampler=valid_sampler)
test_loader = DataLoader(valid_data, batch_size=32, sampler=test_sampler)

# Verify Data Loaders
for inputs, labels in train_loader:
    print(inputs.shape)  # Should print [batch_size, 3, 256, 256]
    break

# Model Definition


class MonkeypoxModel(nn.Module):
    def __init__(self):
        super(MonkeypoxModel, self).__init__()
        self.base_model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Identity()  # Removing the last fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 164)
        self.bn2 = nn.BatchNorm1d(164)
        self.fc3 = nn.Linear(164, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        # Reshape the tensor to have 4 dimensions
        x = x.view(x.size(0), -1, 1, 1)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Instantiate the model, loss function, and optimizer
model = MonkeypoxModel()
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model


def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=5):
    best_accuracy = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to(
                'cuda' if torch.cuda.is_available() else 'cpu').float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.round()
            running_corrects += torch.sum(preds == labels.unsqueeze(1)).item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        model.eval()
        valid_loss = 0.0
        valid_corrects = 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                labels = labels.to(
                    'cuda' if torch.cuda.is_available() else 'cpu').float()

                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))

                valid_loss += loss.item() * inputs.size(0)
                preds = outputs.round()
                valid_corrects += torch.sum(preds ==
                                            labels.unsqueeze(1)).item()

        valid_loss = valid_loss / len(valid_loader.dataset)
        valid_acc = valid_corrects / len(valid_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f}')

        # Early stopping logic
        if valid_acc > best_accuracy:
            best_accuracy = valid_acc
            torch.save(model.state_dict(), 'monkeypox_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping')
            break


# Training the model
train_model(model, criterion, optimizer,
            train_loader, valid_loader, num_epochs=5)
