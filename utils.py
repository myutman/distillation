import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm

def eval_accuracy(model: nn.Module, test_loader: DataLoader):
    device = list(model.state_dict().values())[0].device
    
    model.eval()
    accuracies = []
    for image, label in test_loader:
        output = model(image.to(device))
        classes = output.argmax(dim=-1).cpu()
        accuracies.append((classes == label).float())
    accuracy = float(torch.cat(accuracies).mean())
    return accuracy

cross_entropy = nn.CrossEntropyLoss()

def train_epoch(model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader):
    device = list(model.state_dict().values())[0].device
    
    model.train()
    accuracies = []
    for image, label in tqdm(train_loader):
        output = model(image.to(device))
        loss = cross_entropy(F.softmax(output, dim=-1), label.to(device)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        classes = output.argmax(dim=-1).cpu()
        accuracies.append((classes == label).float())
    accuracy = float(torch.cat(accuracies).mean())
    return accuracy

def discrete_kl(t1, t2, T=10):
    EPS = 1e-8
    t1 = t1 - t1.mean()
    t1 = F.softmax(t1 / T, dim=-1)
    t2 = t2 - t2.mean()
    t2 = F.softmax(t2 / T, dim=-1)
    #print(t1.min(), t2.min())
    return -(t1 * (torch.log(t2 + EPS) - torch.log(t1 + EPS))).sum(dim=-1).mean()

def distillate_epoch(model1: nn.Module, model2: nn.Module, optimizer: optim.Optimizer, transfer_loader: DataLoader):
    device = list(model1.state_dict().values())[0].device
    
    model1.train()
    model2.eval()
    
    accuracies = []
    for image, label in tqdm(transfer_loader):
        output1 = model1(image.to(device))
        output2 = model2(image.to(device))
        loss = cross_entropy(F.softmax(output1, dim = -1), label.to(device)).mean() + discrete_kl(output1, output2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        classes = output1.argmax(dim=-1).cpu()
        accuracies.append((classes == label).float())
    accuracy = float(torch.cat(accuracies).mean())
    return accuracy