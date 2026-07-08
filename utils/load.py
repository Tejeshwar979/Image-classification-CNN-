
import torch 
import torch.nn as nn 
import torch.optim as optim 

import torchvision 
from torchvision import transforms 
from torchvision.datasets import CIFAR10  
from torch.utils.data import  DataLoader  , dataset   


transform = transforms.Compose([
    transforms.ToTensor() , 
    transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5)) 
])

traindata = CIFAR10(root = "./data" , train = True , download = True , transform = transforms)

testdata = CIFAR10(root = "./data" , train = False , download = True , transform = transforms)

trainloader = DataLoader(traindata , shuffle = True , batch_size = 32)

testloader = DataLoader(testdata , shuffle = True , batch_size = 32)


