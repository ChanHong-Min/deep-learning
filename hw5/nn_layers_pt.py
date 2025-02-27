"""
nn_layers_pt.py

PyTorch version of nn_layers
"""

import torch
import torch.nn.functional as F
import numbers
import numpy as np
import math

def view_as_windows(arr_in, window_shape, step=1):
    # ... 기존 함수 내용 그대로 ...
    pass

#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:
    def __init__(self, f_height, f_width, input_size, in_ch_size, out_ch_size):
        # Xavier initialization
        self.W = torch.normal(0, 1 / math.sqrt((in_ch_size * f_height * f_width / 2)),
                              size=(out_ch_size, in_ch_size, f_height, f_width))
        self.b = 0.01 + torch.zeros(size=(1, out_ch_size, 1, 1))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.v_W = torch.zeros_like(self.W)
        self.v_b = torch.zeros_like(self.b)

    def forward(self, x):
        """
        Perform forward pass for the convolutional layer.
        """
        # Using torch.nn.functional.conv2d for forward pass
        output = F.conv2d(x, self.W, bias=self.b, stride=1, padding=0)
        return output

    def step(self, lr, friction):
        """
        Perform gradient descent step for convolutional layer.
        """
        with torch.no_grad():
            self.v_W = friction * self.v_W + (1 - friction) * self.W.grad
            self.v_b = friction * self.v_b + (1 - friction) * self.b.grad
            self.W -= lr * self.v_W
            self.b -= lr * self.v_b

            self.W.grad.zero_()
            self.b.grad.zero_()

class nn_max_pooling_layer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        """
        Perform forward pass for max pooling layer.
        """
        # Using torch.nn.functional.max_pool2d for forward pass
        output = F.max_pool2d(x, kernel_size=self.pool_size, stride=self.stride)
        return output

class nn_activation_layer:
    def __init__(self):
        pass

    def forward(self, x):
        """
        Perform ReLU activation.
        """
        return x.clamp(min=0)

class nn_fc_layer:
    def __init__(self, input_size, output_size, std=1):
        # Xavier/He initialization
        self.W = torch.normal(0, std / math.sqrt(input_size / 2), (output_size, input_size))
        self.b = 0.01 + torch.zeros((output_size))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.v_W = torch.zeros_like(self.W)
        self.v_b = torch.zeros_like(self.b)

    def forward(self, x):
        """
        Perform forward pass for fully connected layer.
        """
        batch_size = x.shape[0]
        Wx = torch.mm(x.view(batch_size, -1), self.W.T)
        out = Wx + self.b
        return out

    def step(self, lr, friction):
        """
        Perform gradient descent step for fully connected layer.
        """
        with torch.no_grad():
            self.v_W = friction * self.v_W + (1 - friction) * self.W.grad
            self.v_b = friction * self.v_b + (1 - friction) * self.b.grad
            self.W -= lr * self.v_W
            self.b -= lr * self.v_b

            self.W.grad.zero_()
            self.b.grad.zero_()

class nn_softmax_layer:
    def __init__(self):
        pass

    def forward(self, x):
        """
        Perform softmax activation.
        """
        s = x - torch.unsqueeze(torch.amax(x, axis=1), -1)
        return torch.exp(s) / torch.unsqueeze(torch.sum(torch.exp(s), axis=1), -1)

class nn_cross_entropy_layer:
    def __init__(self):
        self.eps = 1e-15

    def forward(self, x, y):
        """
        Compute cross-entropy loss.
        """
        batch_size = x.shape[0]
        num_class = x.shape[1]
        
        # Convert y to one-hot encoding
        onehot = torch.zeros(batch_size, num_class)
        onehot[torch.arange(batch_size), y] = 1

        # Avoid numerical instability
        x = torch.clamp(x, min=self.eps)
        x = x / torch.sum(x, axis=1, keepdim=True)

        # Compute cross-entropy loss
        loss = -torch.sum(onehot * torch.log(x)) / batch_size
        return loss
