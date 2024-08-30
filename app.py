from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# Define the model architecture


class MonkeypoxModel(torch.nn.Module):
    def __init__(self):
        super(MonkeypoxModel, self).__init__()
        self.base_model = torch.hub.load(
            'pytorch/vision:v0.13.1', 'resnet50', pretrained=True)
        self.base_model.fc = torch.nn.Identity()
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(2048, 256)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 164)
        self.bn2 = torch.nn.BatchNorm1d(164)
        self.fc3 = torch.nn.Linear(164, 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Load the trained model
model = MonkeypoxModel()
model.load_state_dict(torch.load("monkeypox_model.pth"))
model.eval()

# Define the labels for binary classification
labels = ["Monkey Pox", "Non Monkey Pox"]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', label="No file uploaded")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', label="No file selected")

        # Read the image and preprocess it
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')

        # Apply transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        # Make a prediction and display the result
        with torch.no_grad():
            pred = model(img_tensor)
            label = labels[int(round(pred.item()))]

        return render_template('index.html', label=label, img_data=file.read())

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
