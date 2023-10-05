from collections import OrderedDict
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

#TO DO: Complete this with your CNN architecture. Make sure to complete the architecture requirements.
#The init has in_channels because this changes based on the dataset. 

n_epochs = 5
batch_size_train = 200
learning_rate = 1e-3
validation_interval = 3
log_interval = 100

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.third_in_channels = None
        self.third_out_channels = None
        if self.in_channels == 3:
            self.third_in_channels = 3840
            self.third_out_channels = 70
        else:
            self.third_in_channels = 2940
            self.third_out_channels = 60
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=30, kernel_size=3, padding=1),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=30, out_channels=60, kernel_size=3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),

            nn.Linear(self.third_in_channels, self.third_out_channels),
            nn.ReLU(),

            nn.Linear(self.third_out_channels, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if self.in_channels == 3:
            x = x.view(-1, 3, 32, 32)
        output = self.model(x)
        return output

#Function to get train and validation datasets. Please do not make any changes to this function.
def load_dataset(
        dataset_name: str,
):
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset, valid_dataset = random_split(full_dataset, [48000, 12000])

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    else:
        raise Exception("Unsupported dataset.")

    return train_dataset, valid_dataset

#TO DO: Complete this function. This should train the model and return the final trained model.
#Similar to Assignment-1, make sure to print the validation accuracy to see
#how the model is performing.

def eval(data_loader, model, device, dataset):
    loss = 0
    correct = 0
    with torch.no_grad(): # notice the use of no_grad
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            loss += F.cross_entropy(output, target).item()
    loss /= len(data_loader.dataset)
    print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))

def train(
        epoch,
        data_loader,
        model,
        device,
        optimizer
):
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item())
            )

def CNN(dataset_name, device):
    #CIFAR-10 has 3 channels whereas MNIST has 1.
    in_channels = None
    if dataset_name == "CIFAR10":
        in_channels= 3
    elif dataset_name == "MNIST":
        in_channels = 1
    else:
        raise AssertionError(f'invalid dataset: {dataset_name}')
    model = Net(in_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_dataset, valid_dataset = load_dataset(dataset_name)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size_train, shuffle=False)
    eval(valid_loader, model, device, "Validation")
    for epoch in range(1, n_epochs + 1):
        train(epoch, train_loader, model, device, optimizer)
        if epoch % validation_interval == 0:
            eval(valid_loader, model, device, "Validation")
    results = dict(
        model=model
    )
    return results