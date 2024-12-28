import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint (val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint (val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save (model.state_dict (), 'checkpoint.pt')


def define_model(trial):
    n_layers = trial.suggest_int ("n_layers", 1, 3)
    layers = []

    in_channels = 3
    for i in range (n_layers):
        out_channels = trial.suggest_int (f"n_units_l{i}", 16, 128, log=True)
        layers.append (nn.Conv2d (in_channels, out_channels, kernel_size=3, padding=1))
        layers.append (nn.ReLU ())
        layers.append (nn.MaxPool2d (kernel_size=2, stride=2))
        in_channels = out_channels

    layers.append (nn.Flatten ())
    linear_input_size = out_channels * (32 // (2 ** n_layers)) ** 2
    layers.append (nn.Linear (linear_input_size, 10))
    return nn.Sequential (*layers)


def objective(trial):
    transform = transforms.Compose (
        [transforms.ToTensor (),
         transforms.Normalize ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10 (root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10 (root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader (trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader (testset, batch_size=128, shuffle=False, num_workers=4)

    model = define_model (trial)
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    model.to (device)

    lr = trial.suggest_float ("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical ("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr (optim, optimizer_name) (model.parameters (), lr=lr)
    criterion = nn.CrossEntropyLoss ()

    early_stopping = EarlyStopping (patience=10, delta=0.001)

    for epoch in range (100):
        model.train ()
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm (enumerate (trainloader, 0), total=len (trainloader), desc=f"Epoch {epoch + 1}")

        for i, data in progress_bar:
            inputs, labels = data
            inputs, labels = inputs.to (device), labels.to (device)
            optimizer.zero_grad ()
            outputs = model (inputs)
            loss = criterion (outputs, labels)
            loss.backward ()
            optimizer.step ()

            running_loss += loss.item ()
            _, predicted = torch.max (outputs.data, 1)
            total += labels.size (0)
            correct += (predicted == labels).sum ().item ()

            progress_bar.set_postfix (loss=running_loss / (i + 1), accuracy=100. * correct / total)

        model.eval ()
        val_loss = 0.0
        with torch.no_grad ():
            for data in testloader:
                images, labels = data
                images, labels = images.to (device), labels.to (device)
                outputs = model (images)
                loss = criterion (outputs, labels)
                val_loss += loss.item ()

        val_loss /= len (testloader)
        early_stopping (val_loss, model)

        if early_stopping.early_stop:
            print ("Early stopping")
            break

    model.load_state_dict (torch.load ('checkpoint.pt'))
    correct = 0
    total = 0
    with torch.no_grad ():
        for data in testloader:
            images, labels = data
            images, labels = images.to (device), labels.to (device)
            outputs = model (images)
            _, predicted = torch.max (outputs.data, 1)
            total += labels.size (0)
            correct += (predicted == labels).sum ().item ()
    accuracy = correct / total

    return accuracy


if __name__ == '__main__':
    optuna.logging.set_verbosity (optuna.logging.WARNING)

    study = optuna.create_study (direction="maximize")
    study.optimize (objective, n_trials=50)

    print ("Best trial:")
    trial = study.best_trial
    print ("  Value: ", trial.value)
    print ("  Params: ")
    for key, value in trial.params.items ():
        print (f"    {key}: {value}")

    optuna.visualization.plot_optimization_history (study)
    optuna.visualization.plot_parallel_coordinate (study)
    optuna.visualization.plot_param_importances (study)
