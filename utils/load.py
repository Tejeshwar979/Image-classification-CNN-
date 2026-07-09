import torch 
import torch.nn as nn  
import os 
import torchvision 
from PIL import Image  
from torchvision import transforms 
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageFolder 



transform = transforms.Compose([

    transforms.Resize((256 , 256)) , 
    transforms.ToTensor() , 
    transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5))

])

try:
    train_dataset = ImageFolder(root = r"D:\CNN\pneumonai-images\chest_xray\train", transform = transform)

    test_dataset = ImageFolder(root = r"D:\CNN\pneumonai-images\chest_xray\test"  , transform = transform)

    train_dataloader = DataLoader(train_dataset , shuffle = True , batch_size = 16)
    test_dataloader = DataLoader(test_dataset , shuffle = True , batch_size = 16) 

except Exception as e : 
       print("exception" , e)


print("loaded")
