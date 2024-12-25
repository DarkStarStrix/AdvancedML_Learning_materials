import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Enable benchmark mode in cudnn
torch.backends.cudnn.benchmark = True

# Check if GPU is available
device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')


# Define the neural network architecture
class Net (nn.Module):
    def __init__(self):
        super (Net, self).__init__ ()
        self.fc1 = nn.Linear (784, 128)
        self.relu = nn.ReLU ()
        self.fc2 = nn.Linear (128, 10)

    def forward(self, x):
        x = x.view (-1, 784)
        x = self.fc1 (x)
        x = self.relu (x)
        x = self.fc2 (x)
        return x


def plot_metrics(metrics, title):
    plt.figure (figsize=(10, 5))
    plt.plot (metrics)
    plt.title (title)
    plt.xlabel ('Batch')
    plt.ylabel (title)
    plt.show ()


def main():
    # Load the MNIST dataset
    transform = transforms.Compose ([transforms.ToTensor (), transforms.Normalize ((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST ('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader (train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # Initialize the model and optimizer
    model = Net ().to (device)
    optimizer = optim.Adam (model.parameters (), lr=0.001)
    criterion = nn.CrossEntropyLoss ()

    # Training loop
    metrics = []
    for epoch in range (10):
        model.train ()
        for batch_idx, (data, target) in enumerate (train_loader):
            data, target = data.to (device), target.to (device)
            optimizer.zero_grad ()
            output = model (data)
            loss = criterion (output, target)
            loss.backward ()
            optimizer.step ()

            if batch_idx % 100 == 0:
                print ('Epoch {}, Batch {}, Loss: {:.4f}'.format (epoch, batch_idx, loss.item ()))
                metrics.append (loss.item ())

    plot_metrics (metrics, 'Loss')

    # Function to show model predictions
    def show_model_predictions(model):
        model.eval ()
        test_dataset = datasets.MNIST ('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader (test_dataset, batch_size=64, shuffle=True, num_workers=4)

        correct = 0
        total = 0

        with torch.no_grad ():
            for data, target in test_loader:
                data, target = data.to (device), target.to (device)
                outputs = model (data)
                _, predicted = torch.max (outputs.data, 1)
                total += target.size (0)
                correct += (predicted == target).sum ().item ()

        print ('Accuracy: {:.2f}%'.format (100 * correct / total))

        # Show some images and predictions
        data, target = next (iter (test_loader))
        data, target = data.to (device), target.to (device)
        outputs = model (data)
        _, predicted = torch.max (outputs.data, 1)
        plt.figure (figsize=(10, 10))
        for i in range (9):
            plt.subplot (3, 3, i + 1)
            plt.imshow (data [i].cpu ().numpy () [0], cmap='gray')
            plt.title ('Predicted: {}'.format (predicted [i].item ()))
            plt.axis ('off')
        plt.show ()

    show_model_predictions (model)


if __name__ == '__main__':
    main ()
