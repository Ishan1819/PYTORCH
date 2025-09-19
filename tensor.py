import torch

# Creating two tensors
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Operations
print("Addition:\n", a + b)
print("Multiplication:\n", a * b)
print("Reshaped:\n", a.view(4))
