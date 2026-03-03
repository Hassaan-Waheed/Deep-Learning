import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

class FashionMLP(nn.Module):
    def __init__(self):
        super(FashionMLP,self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x= x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
if __name__ == '__main__':
    model = FashionMLP()
    print(model)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='./root', train=False, download=True, transform=transform)

    train_loader= DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print("Training started...")
    for epoch in range(5):
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
        for X, y in test_loader:
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f'Accuracy on 10,000 test images: {100 * correct / total}%')
    print("Success! MLP is trained.")
    

