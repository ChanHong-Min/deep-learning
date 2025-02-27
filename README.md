# Project Overview

This project consists of multiple Python scripts, each covering different machine learning and deep learning concepts. The roles and implementations of each file are described below.

---

## 📂 File Descriptions

### 1️⃣ [`hw2.py`](hw2/hw2.py)
- **Description:** Solving a multi-class classification problem using a softmax classifier.
- **Key Concepts:** 
  - Cross-entropy loss function
  - Implementation and evaluation of a linear classifier
  - Optimization using `scipy.optimize.minimize`

---

### 2️⃣ [`hw3.py`](hw3.py)
- **Description:** Implementation of fundamental neural network layers.
- **Key Concepts:** 
  - Linear layer (`nn_linear_layer`)
  - Activation function layer (`nn_activation_layer`)
  - Softmax layer (`nn_softmax_layer`)
  - Cross-entropy loss function (`nn_cross_entropy_layer`)
  - Backpropagation and weight updates

---

### 3️⃣ [`hw4.py`](hw4/hw4.py)
- **Description:** Implementation of key components of a Convolutional Neural Network (CNN).
- **Key Concepts:** 
  - `view_as_windows`: Transforming input tensors using sliding windows
  - Convolutional layer (`nn_convolutional_layer`)
  - Max pooling layer (`nn_max_pooling_layer`)

---

### 4️⃣ [`hw5.py`](hw5/hw5.py)
- **Description:** Image classification model using the MNIST dataset.
- **Key Concepts:** 
  - Loading data from `keras.datasets.mnist`
  - Basic CNN model (`nn_mnist_classifier`) and PyTorch-based model (`MNISTClassifier_PT`)
  - Training with Stochastic Gradient Descent (SGD) and cross-entropy loss

---

### 5️⃣ [`hw7.py`](hw7/hw7.py)
- **Description:** Transformer-based sentiment analysis model.
- **Key Concepts:** 
  - Multi-head attention (`MultiHeadAttention`)
  - Transformer encoder block (`TF_Encoder_Block`)
  - Positional encoding (`PosEncoding`)
  - `sentiment_classifier`: Sentiment analysis model using the IMDB dataset
