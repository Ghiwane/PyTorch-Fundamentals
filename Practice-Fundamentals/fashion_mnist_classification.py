import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import numpy as np

# Load the FashionMNIST training dataset and convert images to PyTorch tensors
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# Load the FashionMNIST test dataset
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# Create DataLoaders to process datasets in batches of 64 (shuffle training data for randomness)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Define a Multi-Layer Perceptron (MLP) neural network architecture
class Fashion_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),          # Flatten the 28x28 image tensors into 784 1D vectors
            nn.Linear(784, 128),   # First fully connected layer (784 inputs -> 128 outputs)
            nn.ReLU(),             # ReLU activation function
            nn.Dropout(0.2),       # Dropout layer (randomly zeros 20% of activations to reduce overfitting)
            nn.Linear(128, 64),    # Second fully connected layer (128 -> 64)
            nn.ReLU(),             # ReLU activation function
            nn.Linear(64, 10)      # Output layer (64 -> 10 classes)
        )
    def forward(self, X):
        return self.net(X)

# Set manual seed for reproducibility of random initializations
torch.manual_seed(17)
Fashion = Fashion_model()

# Define CrossEntropyLoss for multi-class classification and Adam optimizer with lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Fashion.parameters(), lr=0.001)

# Training parameters and lists to track metrics over epochs
epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# Track start time for benchmarking training duration
start_time = time.time()

for e in range(epochs):
    Fashion.train() # Set model to training mode (enables Dropout)

    correct_predictions_train = 0
    correct_predictions_test= 0
    total_samples = 0
    running_loss = 0.0

    # Training loop over batches
    for images, labels in train_loader:
        optimizer.zero_grad()           # Reset gradients from previous step
        logits_pred = Fashion(images)   # Forward pass
        loss = criterion(logits_pred, labels) # Compute batch loss
        loss.backward()                 # Backward pass (compute gradients)
        optimizer.step()                # Update network weights

        running_loss += loss.item() * labels.size(0)

        # Calculate training accuracy
        predictions = torch.argmax(logits_pred, dim=1)
        correct_predictions_train += (predictions == labels).sum().item()

        total_samples += labels.size(0)

    # Compute overall training metrics for the current epoch
    epoch_loss = running_loss / total_samples
    epoch_accuracy = (correct_predictions_train / total_samples) * 100
    
    print(f"Epoch train [{e+1}/{epochs}] "
        f"| Loss train : {epoch_loss:.4f} "
        f"| Accuracy train : {epoch_accuracy:.2f}%")

    train_losses.append(epoch_loss)
    train_correct.append(correct_predictions_train)

    Fashion.eval() # Set model to evaluation mode (disables Dropout)
    running_loss_test=0.0
    correct_predictions_test=0
    total_test_samples = 0
    with torch.no_grad(): # Disable gradient computation during testing for speed and memory saving
        for X_test, y_test in test_loader:
            y_eval = Fashion(X_test)
            predictions = torch.argmax(y_eval, dim=1)
            correct_predictions_test += (predictions == y_test).sum().item()
            loss = criterion(y_eval, y_test)
            running_loss_test += loss.item() * y_test.size(0)
            total_test_samples += y_test.size(0)

    # Compute overall test metrics for the current epoch
    epoch_loss_test = running_loss_test / total_test_samples
    epoch_accuracy_test = (correct_predictions_test / total_test_samples) * 100
        
    print(f"Epoch test [{e+1}/{epochs}] "
        f"| Loss test : {epoch_loss_test:.4f} "
        f"| Accuracy test : {epoch_accuracy_test:.2f}%")        
    test_losses.append(epoch_loss_test)
    test_correct.append(correct_predictions_test)

# Display total elapsed execution time
current_time = time.time()
total_time = (current_time - start_time)/60
print(f'it took {total_time:.2f} minutes to train!')

# Plot training vs test loss curves
plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Test loss")
plt.title("Loss at epoch")
plt.legend()
plt.show()

# Plot training vs test accuracy curves (600 = 60000/100, 100 = 10000/100)
plt.plot([t/600 for t in train_correct], label="Training accuracy")
plt.plot([t/100 for t in test_correct], label="Test accuracy")
plt.title("Accuracy at the end of each epoch")
plt.legend()
plt.show()