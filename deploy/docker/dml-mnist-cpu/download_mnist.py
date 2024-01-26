from torchvision import datasets, transforms

def download():
    # Load MNIST dataset.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    datasets.MNIST('data', train=True, download=True, transform=transform)
    print("Downloaded MNIST dataset")


if __name__ == "__main__":
    download()

