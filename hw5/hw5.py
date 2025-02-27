"""
HW5
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import importlib
import pickle

import torch
import torch.nn as nn
import math

from keras.datasets import mnist

import nn_layers_pt as nnl
importlib.reload(nnl)

########################
#   Your custom classifier
#   based on previous homeworks
########################

class nn_mnist_classifier:
    def __init__(self, mmt_friction=0.9, lr=1e-2):
        ## initialize each layer of the entire classifier

        # convolutional layer
        self.conv_layer_1 = nnl.nn_convolutional_layer(f_height=3, f_width=3, input_size=28,
                                                       in_ch_size=1, out_ch_size=28)
        # activation: relu
        self.act_1 = nnl.nn_activation_layer()
        # maxpool
        self.maxpool_layer_1 = nnl.nn_max_pooling_layer(pool_size=2, stride=2)
        # fully connected layers
        self.fc1 = nnl.nn_fc_layer(input_size=28*13*13, output_size=128)
        self.act_2 = nnl.nn_activation_layer()
        self.fc2 = nnl.nn_fc_layer(input_size=128, output_size=10)
        # softmax
        self.sm1 = nnl.nn_softmax_layer()
        # cross entropy
        self.xent = nnl.nn_cross_entropy_layer()

        # learning rate and momentum
        self.lr = lr
        self.mmt_friction = mmt_friction

    def forward(self, x, y):
        cv1_f = self.conv_layer_1.forward(x)
        ac1_f = self.act_1.forward(cv1_f)
        mp1_f = self.maxpool_layer_1.forward(ac1_f)

        mp1_f_flat = mp1_f.reshape(mp1_f.shape[0], -1)
        fc1_f = self.fc1.forward(mp1_f_flat)
        ac2_f = self.act_2.forward(fc1_f)
        fc2_f = self.fc2.forward(ac2_f)

        sm1_f = self.sm1.forward(fc2_f)
        cn_f = self.xent.forward(sm1_f, y)

        scores = sm1_f
        loss = cn_f
        return scores, loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def step(self):
        self.conv_layer_1.step(self.lr, self.mmt_friction)
        self.fc1.step(self.lr, self.mmt_friction)
        self.fc2.step(self.lr, self.mmt_friction)

########################
#   Classifier based on PyTorch modules
########################

class MNISTClassifier_PT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Conv2d(1, 28, kernel_size=3)
        self.act_1 = nn.ReLU()
        self.maxpool_layer_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(28 * 13 * 13, 128)
        self.act_2 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        cv1_f = self.conv_layer_1(x)
        ac1_f = self.act_1(cv1_f)
        mp1_f = self.maxpool_layer_1(ac1_f)

        mp1_f_flat = mp1_f.view(mp1_f.size(0), -1)
        fc1_f = self.fc1(mp1_f_flat)
        ac2_f = self.act_2(fc1_f)
        out_logit = self.fc2(ac2_f)
        return out_logit

########################
## classification: dataset preparation
########################

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    X_train = X_train.astype('float64') / 255.0
    X_test = X_test.astype('float64') / 255.0

    n_train_sample = 50000
    X_val, X_train = np.split(X_train, [len(y_train) - n_train_sample])
    y_val, y_train = np.split(y_train, [len(y_train) - n_train_sample])

    trn_dataset = [(d, l) for d, l in zip(X_train, y_train)]
    val_dataset = [(d, l) for d, l in zip(X_val, y_val)]
    test_dataset = [(d, l) for d, l in zip(X_test, y_test)]

    lr = 0.01
    n_epoch = 2
    batch_size = 128
    friction = 0.9

    PYTORCH_BUILTIN = True
    if PYTORCH_BUILTIN:
        classifier = MNISTClassifier_PT()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=friction)
    else:
        classifier = nn_mnist_classifier(mmt_friction=friction, lr=lr)

    train_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True)

    for i in range(n_epoch):
        trn_accy = 0
        for X, y in train_loader:
            X, y = torch.tensor(X), torch.tensor(y).long()
            if PYTORCH_BUILTIN:
                scores = classifier(X)
                loss = criterion(scores, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                scores, loss = classifier(X, y)
                classifier.step()

            estim = torch.argmax(scores, axis=1)
            trn_accy += (estim == y).float().mean().item()

        print(f"Epoch {i + 1}, Training Accuracy: {trn_accy / len(train_loader):.4f}")

    tot_accy = 0
    for X, y in test_loader:
        X, y = torch.tensor(X), torch.tensor(y).long()
        with torch.no_grad():
            scores = classifier(X) if PYTORCH_BUILTIN else classifier(X, y)[0]
            tot_accy += (torch.argmax(scores, axis=1) == y).float().mean().item()

    print(f"Test Accuracy: {tot_accy / len(test_loader):.4f}")
