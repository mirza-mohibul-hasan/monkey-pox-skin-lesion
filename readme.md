# Monkeypox Skin Lesion Detection

This project is a deep learning-based web application designed to detect Monkeypox skin lesions from images. It utilizes a ResNet-50 model, fine-tuned for binary classification to distinguish between Monkeypox and non-Monkeypox skin lesions.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Testing Accuracy](#testing-accuracy)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The main objective of this project is to build a reliable tool that can assist in the early detection of Monkeypox by analyzing skin lesion images. The model is trained on a dataset of augmented images and validated on original images to ensure accuracy and generalization.

## Project Structure

The project directory is structured as follows:

```plaintext
MONKEY POX SKIN LESION
│
├── Dataset/
│   ├── Augmented Images/
│   └── Original Images/
│
├── templates/
│   └── index.html
│
├── accuracy.py
├── app.py
├── model.py
├── monkeypox_model.pth
└── requirements.txt
```

- Dataset/: Contains the dataset used for training and validation.
- templates/: Contains the HTML template for the web application.
- accuracy.py: Script to evaluate the model's accuracy on test data.
- app.py: Flask web application to serve the model and allow users to upload images for prediction.
- model.py: Python script to define the deep learning model and train it.
- monkeypox_model.pth: Pre-trained model weights.
- requirements.txt: List of dependencies required to run the project.

## Requirements

Ensure you have the following installed:

- Python 3.7 or higher
- pip package manager
  The required Python packages can be installed using:

```bash
pip install -r requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/monkeypox-skin-lesion-detection.git
cd monkeypox-skin-lesion-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download pre-trained model weights:

```plaintext
If you want to skip training, you can use the provided monkeypox_model.pth file.
```

## Usage

### Running the Web Application

To start the web application:

```bash
python app.py
```

This will launch the Flask application, accessible via http://127.0.0.1:5000 in your web browser. You can upload an image to the application, and it will predict whether the lesion in the image is indicative of Monkeypox.

### Model Training

If you want to train the model from scratch:

```bash
python model.py
```

This script will train the model using the dataset in the Dataset/ directory. The best model will be saved as monkeypox_model.pth.

### Testing Accuracy

To evaluate the model's accuracy on the test dataset:

```bash
python accuracy.py
```

This script will load the model and compute its accuracy on the test dataset, outputting the loss and accuracy metrics.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
