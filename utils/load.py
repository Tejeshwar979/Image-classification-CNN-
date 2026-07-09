import torch 
import torch.nn as nn  
import os 
import torchvision 
from PIL import Image  
from torchvision import transforms 
from torch.utils.data import DataLoader 



class image_processing : 
    def __init__(self , root_dir_path):
        self.root_dit_path = root_dir_path 
        
        self.image_dataset = [os.path.join(img , root_dir_path) for img in os.listdir(root_dir_path)] 

        self.transformations = transforms.Compose([
            transforms.ToTensor() , 
            transforms.Normalize((0.5 , 0.5 , 0.5) , (0.5 , 0.5 , 0.5) )
        ])

    def __len__(self):
        return len(self.image_dataset)
        
    def __getitem__(self, idx):
        img_path , label = self.image_dataset[idx]
        img = Image.open(img_path).convert("RGB")
        return img , label   

train_dataset = image_processing(root_dir_path="./train_data")
test_dataset =  image_processing(root_dir_path='./test_data')

train_dataloader = DataLoader(train_dataset , shuffle = True , batch_size = 32)
test_dataloader = DataLoader(test_dataset , shuffle = True , batch_size=32)


