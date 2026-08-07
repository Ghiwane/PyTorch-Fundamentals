import torch
import torch.nn as nn

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