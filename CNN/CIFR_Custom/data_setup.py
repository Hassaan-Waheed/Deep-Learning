import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def get_loaders(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("Downloading and loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    print("Data loaders created successfully.")

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"Classes: {classes}")
    return train_loader, test_loader, classes

def visualize_samples(loader, classes):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    print("Visualizing sample images from the dataset...")

    # Unnormalize for viewing
    img_grid = torchvision.utils.make_grid(images[:4]) / 2 + 0.5
    npimg = img_grid.numpy()

    print("Normalized images converted to numpy for visualization.")
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(' '.join(f'{classes[labels[j]]}' for j in range(4)))
    plt.savefig('cifar_samples.png') 
    print("Saved samples to cifar_samples.png")
    #plt.show()

if __name__ == "__main__":
    # Test if it works
    loader, _, classes = get_loaders()
    visualize_samples(loader, classes)