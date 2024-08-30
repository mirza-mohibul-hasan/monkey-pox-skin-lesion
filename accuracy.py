import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Define the transformations to be applied to the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the test data
test_data = datasets.ImageFolder(
    './Dataset/Original Images', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Define the PyTorch model


class MonkeypoxModel(nn.Module):
    def __init__(self):
        super(MonkeypoxModel, self).__init__()
        self.base_model = models.resnet50(pretrained=False)
        self.base_model.fc = nn.Identity()  # Remove the fully connected layer
        self.fc1 = nn.Linear(2048, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 164)
        self.bn2 = nn.BatchNorm1d(164)
        self.fc3 = nn.Linear(164, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
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


# Determine the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model and move it to the appropriate device
model = MonkeypoxModel()
model.load_state_dict(torch.load('monkeypox_model.pth', map_location=device))
model.to(device)  # Move the model to the device
model.eval()

# Define the loss function and optimizer
criterion = nn.BCELoss()

# Evaluate the model on the test data
test_loss = 0.0
test_corrects = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)  # Move inputs to the device
        labels = labels.to(device).float()  # Move labels to the device

        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))

        test_loss += loss.item() * inputs.size(0)
        preds = outputs.round()
        test_corrects += torch.sum(preds == labels.unsqueeze(1)).item()

test_loss = test_loss / len(test_loader.dataset)
test_accuracy = test_corrects / len(test_loader.dataset)
print("Test Loss:", test_loss * 100, "%")
print("Test Accuracy:", test_accuracy * 100, "%")
