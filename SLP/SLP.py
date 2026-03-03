import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x.view(-1, 784))

if __name__ == '__main__':
    # Standard MNIST setup
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # num_workers=0 is the key for stability on Windows
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    model = SLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("Training started...")
    for epoch in range(2):
        for batch, (X, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            if batch % 200 == 0:
                print(f"Epoch {epoch+1}, Batch {batch}, Loss: {loss.item():.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in DataLoader(test_data, batch_size=64, num_workers=0):
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f'Accuracy on 10,000 test images: {100 * correct / total}%')
    print("Success! SLP is trained.")