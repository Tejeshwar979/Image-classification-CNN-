import torch 

import torch.nn as nn  

class CNN(nn.Module) : 
    def __init__(self):
        super(self , CNN).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3 , 32 , 3 , 1),
            nn.ReLU() , 
            nn.MaxPool(2),
            nn.Conv2d(32 , 64 , 3 , 1) ,
            nn.ReLU(),
            nn.MaxPool(2),
            nn.Conv2d(64 , 128 , 3 , 1) , 
            nn.ReLU(),
            nn.MaxPool(2),
            nn.Conv2d(128 , 256 , 3 , 1),
            nn.ReLU(),
            nn.MaxPool(2),
            nn.Conv2d(256 , 512 , 3 , 1),
            nn.ReLU() ,
            nn.MaxPool(2),
            nn.Conv2d(512 , 1024 , 3 , 1),
            nn.ReLU(),
            nn.MaxPool(2) ,
            nn.Flatten()
        )
        pass 


