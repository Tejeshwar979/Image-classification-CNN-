import torch 
from utils.load import * 
import torch.nn as nn  



class CNN(nn.Module) : 
    def __init__(self):
        super(CNN , self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(3 , 32 , 3 , 1),
            nn.ReLU() , 
            nn.MaxPool2d(2 , 2),

            nn.Conv2d(32 , 64 , 3 , 1) ,
            nn.ReLU(),
            nn.MaxPool(2 , 2),

            nn.Conv2d(64 , 128 , 3 , 1) , 
            nn.ReLU(),
            nn.MaxPool(2 , 2),

            nn.Conv2d(128 , 256 , 3 , 1),
            nn.ReLU(),
            nn.MaxPool(2 , 2),
        ) 
        self.fc_layer = nn.Sequential(
            nn.Linear(4 * 4 * 128 , 256), 
            nn.ReLU() , 
            nn.Linear(2 , 2)
        )
    def forward(self , x):
        x = self.model(x) 
        x = x.view(x.shape[0] ,-1)
        return self.fc_layer(x)


