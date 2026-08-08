import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import numpy as np

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class Fashion_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, X):
        return self.net(X)

torch.manual_seed(17)
Fashion = Fashion_model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Fashion.parameters(), lr=0.01)

epochs = 20
train_losses = []
test_losses = []
train_correct = []
test_correct = []

start_time = time.time()

for e in range(epochs):
    Fashion.train()

    correct_predictions_train = 0
    correct_predictions_test= 0
    total_samples = 0
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()           
        logits_pred = Fashion(images)         
        loss = criterion(logits_pred, labels)
        loss.backward()                 
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(logits_pred, dim=1)
        correct_predictions_train += (predictions == labels).sum().item()

        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = (correct_predictions_train / total_samples) * 100
    
    print(f"Epoch [{e+1}/{epochs}] "
        f"| Loss : {epoch_loss:.4f} "
        f"| Accuracy : {epoch_accuracy:.2f}%")

    train_losses.append(loss)
    train_correct.append(correct_predictions_train)

    Fashion.eval()
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_eval = Fashion(X_test)
            predictions = torch.argmax(y_eval, dim=1)
            correct_predictions_test += (predictions == y_test).sum().item()

    loss = criterion(y_eval, y_test)
    test_losses.append(loss)
    test_correct.append(correct_predictions_test)

current_time = time.time()
total_time = (current_time - start_time)/60
print(f'it took {total_time:.2f} minutes to train!')

train_losses = [tl.item() for tl in train_losses]
test_losses = [tl.item() for tl in test_losses]


plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Test loss")
plt.title("Loss at epoch")
plt.legend()
plt.show()

plt.plot([t/600 for t in train_correct], label="Training accuracy")
plt.plot([t/100 for t in test_correct], label="Test accuracy")
plt.title("Accuracy at the end of each epoch")
plt.legend()
plt.show()