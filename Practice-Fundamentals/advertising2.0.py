import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Define a linear regression model class inheriting from PyTorch's nn.Module
class LinearModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # Define a single linear layer mapping n_features input dimensions to 1 output dimension
        self.linear = nn.Linear(n_features, 1)

    # Define the forward pass logic
    def forward(self, x):
        return self.linear(x)

# Build an absolute path to the CSV file so the script works reliably across different operating systems
current_folder = os.path.dirname(os.path.abspath(__file__))
path_csv = os.path.join(current_folder, "data", "advertising.csv")
df = pd.read_csv(path_csv)

# Extract feature columns (X) and target variable (y) as NumPy arrays
df["tv_x_radio"] = df["tv"] * df["radio"]
X = df[["tv", "radio", "journaux", "tv_x_radio"]].values
y = df["ventes"].values

# Split dataset into training (80%) and testing (20%) sets using NumPy arrays
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize feature scaler to standardize input values (mean=0, std=1)
scaler = StandardScaler()

# Fit scaler on training set and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert standardized NumPy arrays into PyTorch float32 tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Reshape target variables from (N,) to column vectors (N, 1) to match model output shape
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Retrieve the number of input features (3 columns) and instantiate the model
n_features = X_train_tensor.shape[1]
model = LinearModel(n_features)

# Define Mean Squared Error loss function and Adam optimizer with a learning rate of 0.1
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Training loop configuration
epochs = 500
losses = []

# Main training loop over epochs
for e in range(epochs):
    # Reset accumulated gradients from previous iteration
    optimizer.zero_grad()
    
    # Forward pass: compute predictions for training data
    y_pred = model(X_train_tensor)
    
    # Compute loss between predictions and true target values
    loss = loss_fn(y_pred, y_train_tensor)
    
    # Backward pass: calculate gradients via automatic differentiation
    loss.backward()
    
    # Update model parameters based on gradients
    optimizer.step()

    # Track numerical loss value
    losses.append(loss.item())
    
    # Print progress every 200 epochs
    if e  % 20 == 0:
        print(f"Epoch {e} | Loss (MSE): {loss.item():.4f}")

# Switch model to evaluation mode
model.eval()

# Disable gradient calculation during inference for efficiency
with torch.no_grad():
    # Predict outcomes on test dataset
    y_pred_test = model(X_test_tensor)
    
    # Detach tensor from computational graph and convert back to NumPy array
    y_pred_test = y_pred_test.detach().numpy()

# Calculate performance evaluation metrics (RMSE and R2 score)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

# Display evaluation results
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Plot loss curve over epochs to visualize model convergence
plt.plot(range(epochs), losses)
plt.show()