import torch

def custom_activation(x):
    # Define your custom activation function here
    return x**2 + torch.sin(x)

# Define a tensor with requires_grad=True
x = torch.tensor(0.0, requires_grad=True)

# Apply your custom activation function
y = custom_activation(x)

# Calculate gradients
y.backward()

# Access the gradient of x
grad = x.grad

print("Gradient:", grad.item())

