import torch
from skimage.util import view_as_windows


x = torch.arange(9, dtype=torch.double).reshape(3, 3) + 1
print('x.shape', x.shape)
print('x')
print(x)

# filter
filt = torch.ones((2, 2))
filt = filt.reshape((-1, 1))
print('reshaped filt', filt)

y = view_as_windows(x, (2, 2))
print('y.shape', y.shape)
print('y')
print(y)

y = y.reshape((2, 2, -1))
print('reshaped y')
print(y)

# perform convolution by matrix multiplication
print('convolution result:')
print('y, filt size', y.shape, filt.shape)
result = torch.matmul(y, filt)
result = torch.squeeze(result, axis=2)
print('result is:', result)


