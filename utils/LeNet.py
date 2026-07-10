import torch 
import torch.nn as nn  

class LeNet(nn.Module):
    def __init__(self) :
        super(LeNet , self).__init__() 
        self.model = nn.Sequential(
            nn.Conv2d(3 , 6 , 3 , 1) ,
            nn.tanh(-1 , 1) , 
            nn.AvgPool2d(2 , 2) , 

            nn.Conv2d(6 , 16 , 3 , 1),
            nn.tanh(-1 , 1),
            nn.AvgPool2d(2 , 2) 
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(4 * 4 * 16 , 256) , 
            nn.tanh() ,
            nn.Linear(256 , 1)
        )
    def forward(self , x):
        x = self.model(x)
        x = x.view(x.shape[0] , -1)
        return self.fc_layer(x) 


