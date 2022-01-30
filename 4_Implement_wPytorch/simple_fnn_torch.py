from ctypes.wintypes import tagMSG
from posixpath import dirname
from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

def mnist_data(batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root='0_dataset/MNIST', train=True, download=False,
            # transform=transforms.Compose([
            # transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            # ])
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root='0_dataset/MNIST', train=False, 
            # transform = transforms.Compose([
            #             transforms.ToTensor(),
            #             transforms.Normalize((0.1307,), (0.3081,))])
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

class FNN(nn.Module):
    def __init__(self) -> None:
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(784, 30)
        self.fc2 = nn.Linear(30, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return F.log_softmax(x,dim=1)
    
def train(model, train_loader, optimizer, epoch, device, logging):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx%logging == 0:
            print("epoch:{}, batch_idx:{}, loss:{}".format(epoch,batch_idx,loss.item()))

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__': 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    lr = 5.0
    batch_size = 64
    epochs = 3
    logging = 100
    train_loader, test_loader= mnist_data(batch_size)
    model = FNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(0, epochs):
        train(model, train_loader, optimizer, epoch, device, logging)
        test(model, test_loader, device)
        
