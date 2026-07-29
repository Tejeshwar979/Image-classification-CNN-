import torch 
import torch.nn as nn  



class VGG(nn.Module): 
    def __init__(self):
        super().__init__()
        self.first_convlayers = nn.Sequential(
            nn.Conv2d(3 , 64 , 3 , 1 , 1) , 
            nn.ReLU() , 
            nn.Conv2d(64 , 64 , 3 , 1 , 1),
            nn.ReLU() , 
            nn.MaxPool2d(2 , 2) , 
        )
        self.second_convlayers = nn.Sequential(
            nn.Conv2d(64 , 128 , 3 , 1 , 1) , 
            nn.ReLU(),
            nn.Conv2d(128 , 128 , 3 , 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2 , 2),
        )
        self.third_convlayers = nn.Sequential(
            nn.Conv2d(128 , 256 , 3 , 1 , 1) , 
            nn.ReLU(),
            nn.Conv2d(256 , 256 , 3 , 1, 1),
            nn.ReLU(),
            nn.Conv2d(256 , 256 , 3 , 1 , 1) , 
            nn.ReLU(),
            nn.Conv2d(256 , 256 , 3 , 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2 , 2),
        ) 
        self.fourth_convlayers = nn.Sequential(
            nn.Conv2d(256 , 512 , 3 , 1 , 1) , 
            nn.ReLU(),
            nn.Conv2d(512 , 512 , 3 , 1, 1),
            nn.ReLU(),
            nn.Conv2d(512 , 512 , 3 , 1 , 1) , 
            nn.ReLU(),
            nn.Conv2d(512 , 512 , 3 , 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2 , 2),
        )
        self.fifth_convlayers = nn.Sequential(
            nn.Conv2d(512 , 512 , 3 , 1 , 1) , 
            nn.ReLU(),
            nn.Conv2d(512 , 512 , 3 , 1, 1),
            nn.ReLU(),
            nn.Conv2d(512 , 512 , 3 , 1 , 1) , 
            nn.ReLU(),
            nn.Conv2d(512 , 512 , 3 , 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2 , 2),
        ) 
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 8 * 512 , 4096) , 
            nn.ReLU(),
            nn.Linear(4096 , 1000),
            nn.ReLU(),
            nn.Linear(1000 , 2) 
        )  
    def forward(self , x) : 
        x = self.first_convlayers(x) 
        x = self.second_convlayers(x)
        x = self.third_convlayers(x)
        x = self.fourth_convlayers(x)
        x = self.fifth_convlayers(x)
        x = x.view(x.shape[0] ,-1)
        return self.fc_layers(x)


