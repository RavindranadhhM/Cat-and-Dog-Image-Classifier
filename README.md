# 🐱🐶 Cat and Dog Image Classifier

A Convolutional Neural Network (CNN)–based image classification model built as part of the FreeCodeCamp Machine Learning with Python certification project. The model classifies images as either cat or dog with a test accuracy of 74.0%, successfully passing the certification challenge.
# 📌 Project Overview

This project demonstrates the use of deep learning and computer vision to solve a binary image classification problem. Using TensorFlow/Keras, we trained a CNN from scratch to identify cats and dogs from images. The model architecture consists of multiple convolutional and pooling layers, followed by dense layers for classification.
# 🛠️ Model Architecture

Model type: Sequential CNN
Layers:
1) Conv2D (32 filters, 3×3 kernel) + ReLU activation
2) MaxPooling2D (2×2)
3) Conv2D (64 filters, 3×3 kernel) + ReLU activation
4) MaxPooling2D (2×2)
5) Conv2D (128 filters, 3×3 kernel) + ReLU activation
6) MaxPooling2D (2×2)
7) Flatten
8) Dense (512 units) + ReLU activation
9) Dropout (50%) for regularization
10) Dense (1 unit) + Sigmoid activation (binary classification output)

Total Parameters: 19,034,177 (all trainable)
Optimizer: Adam
Loss Function: Binary Crossentropy
Evaluation Metric: Accuracy
# 📊 Training Performance
* Final Test Accuracy: 74.0% ✅
* Training Accuracy: Reached ~70%
* Validation Accuracy: Peaked slightly above 70%
* Loss Trend: Both training and validation loss decreased steadily, showing no significant overfitting.


<img width="689" height="682" alt="image" src="https://github.com/user-attachments/assets/7f667f6a-911c-4e7f-853e-481cb5a44608" />

# 📂 Dataset
Source: https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip
Provided by FreeCodeCamp for the project.

Classes: `cats` (label 0), `dogs` (label 1)

Preprocessing:
* Images resized to 150×150 pixels
* Normalized pixel values to range [0, 1]
* Augmentation not used (baseline model)

# 🚀 How to Run

You can run the project in Google Colab or locally with Jupyter Notebook:

1. Open `cat_dog.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter.
2. Run all cells to train and test the model.

# 📌 Possible Improvements
* Add data augmentation to improve generalization.
* Use transfer learning (e.g., VGG16, ResNet) to boost accuracy.
* Implement early stopping and learning rate scheduling.
* Optimize batch size and learning rate via hyperparameter tuning.

# 📜 License

This project is part of FreeCodeCamp’s Machine Learning with Python curriculum and is licensed under the MIT License.
