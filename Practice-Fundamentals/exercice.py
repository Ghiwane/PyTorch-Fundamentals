import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import time

# Define a custom neural network module by inheriting from nn.Module
class my_model(nn.Module) :
    def __init__(self, inputs=4, layer=32, outputs=2):
        super().__init__()
        # Define the network architecture using a sequential container
        self.net = nn.Sequential(         
            nn.Linear(inputs, layer),  # Fully connected layer from input to hidden layer (4 -> 32)
            nn.ReLU(),                  # Activation function
            nn.Linear(layer, outputs)   # Fully connected layer from hidden to output layer (32 -> 2)
        )

    # Define the forward pass logic (how data flows through the model)
    def forward(self, x):
        return self.net(x)

# Set seed for reproducible results
torch.manual_seed(7)

# ==========================================
# 1. CPU EXECUTION
# ==========================================
device_cpu = torch.device("cpu")

model_cpu = my_model().to(device_cpu)
x_tensor_cpu = torch.rand(1, 4).to(device_cpu)
new_x_cpu = model_cpu(x_tensor_cpu)

X_cpu = torch.rand(1000000, 4).to(device_cpu)
y_cpu = torch.rand(1000000, 2).to(device_cpu)

loss_fn = nn.MSELoss()
optimizer_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.01)

epochs = 100
losses_cpu = []

# Start CPU timer
start_cpu = time.perf_counter()

for e in range(epochs):
    optimizer_cpu.zero_grad()
    y_pred = model_cpu(X_cpu)
    loss = loss_fn(y_pred, y_cpu)
    loss.backward()
    optimizer_cpu.step()
    losses_cpu.append(loss.item())

end_cpu = time.perf_counter()
cpu_time = end_cpu - start_cpu

# ==========================================
# 2. GPU EXECUTION
# ==========================================
if torch.cuda.is_available():
    device_gpu = torch.device("cuda")

    model_gpu = my_model().to(device_gpu)
    x_tensor_gpu = torch.rand(1, 4).to(device_gpu)
    new_x_gpu = model_gpu(x_tensor_gpu)

    X_gpu = torch.rand(1000000, 4).to(device_gpu)
    y_gpu = torch.rand(1000000, 2).to(device_gpu)

    optimizer_gpu = torch.optim.Adam(model_gpu.parameters(), lr=0.01)

    losses_gpu = []

    # CUDA sync before timing
    torch.cuda.synchronize()
    start_gpu = time.perf_counter()

    for e in range(epochs):
        optimizer_gpu.zero_grad()
        y_pred = model_gpu(X_gpu)
        loss = loss_fn(y_pred, y_gpu)
        loss.backward()
        optimizer_gpu.step()
        losses_gpu.append(loss.item())

    # Wait for GPU computation to finish
    torch.cuda.synchronize()
    end_gpu = time.perf_counter()
    gpu_time = end_gpu - start_gpu

    print(f"Time on CPU: {cpu_time:.6f} seconds")
    print(f"Time on GPU: {gpu_time:.6f} seconds")
else:
    print(f"Time on CPU: {cpu_time:.6f} seconds")
    print("CUDA is not available on this machine.")

# Plot loss curve over epochs to verify convergence
plt.plot(range(epochs), losses_cpu)
plt.show()