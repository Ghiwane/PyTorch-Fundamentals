import torch
import torch.nn as nn
import matplotlib.pyplot as plt 

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

# Instantiate the model using default parameters
model = my_model()

# Create a random input tensor of shape (batch_size=1, features=4)
x_tensor = torch.rand(1, 4)
print(x_tensor,"\n\n")

# Pass the input tensor through the model (automatically calls forward())
new_x = model(x_tensor)

# Print the predicted output tensor and its dimensions
print(new_x, new_x.shape)

# Generate synthetic dataset: 100 samples with 4 features for X, 2 target outputs for y
X = torch.rand(100, 4)
y = torch.rand(100, 2)

# Define Mean Squared Error loss function
loss_fn = nn.MSELoss()

# Instantiate Adam optimizer to update model parameters with learning rate of 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Set training configuration
epochs = 100
losses=[]

# Training loop over 100 epochs
for e in range(epochs):
    # Reset gradients accumulated from the previous iteration
    optimizer.zero_grad()

    # Forward pass: compute predictions for all input samples
    y_pred = model(X)

    # Compute loss between model predictions and ground truth targets
    loss = loss_fn(y_pred, y)

    # Backward pass: compute gradients via backpropagation
    loss.backward()

    # Update model weights based on computed gradients
    optimizer.step()

    # Print current loss every 10 epochs
    if e % 10 == 0 :
        print(f"Epoch {e}, Loss: {loss.item()}")
    
    # Store numerical loss value for visualization
    losses.append(loss.item())

# Plot loss curve over epochs to verify convergence
plt.plot(range(epochs), losses)
plt.show()