import torch

# Enabling gradient tracking
x = torch.tensor(3.0, requires_grad=True)

# Function y = x^2 + 2x
y = x**2 + 2*x

# Compute gradient
y.backward()

print("Value of y:", y.item())
print("dy/dx:", x.grad.item())
