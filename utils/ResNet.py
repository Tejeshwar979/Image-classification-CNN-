import torch 
import torch.nn as nn  



class blocks(nn.Module):
    def __init__(self , in_channels , out_channels , stride = 1 , padding = 1) : 
        super().__init__()
        self.conv1layer = nn.Conv2d(in_channels , out_channels , 3 , stride = stride , padding = 1 )
        self.batch1 = nn.BatchNorm2d(out_channels)
        self.Relu_fnx = nn.ReLU()
        self.conv2layer = nn.Conv2d(out_channels , out_channels , 3 , stride = 1 , padding = 1)
        self.batch2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential() 

        if( (stride != 1) or (in_channels != out_channels)  ): 
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels , out_channels , 1 ,  stride = stride , padding = 0 ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self , x):
        out = self.conv1layer(x)
        out = self.batch1(out)
        out = self.Relu_fnx(out)
        out = self.conv2layer(out)
        out = self.batch2(out)
        shortcut = self.shortcut(x)
        value = out + shortcut
        return self.Relu_fnx(value)

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3 , 64 , 7 , stride = 2 , padding = 3 ) 
        self.bn1 = nn.BatchNorm2d(64)
        self.Relu_fnx = nn.ReLU()
        self.pooling = nn.MaxPool2d(3 , 2 , padding = 1)

        self.layer1block1 = blocks(64 , 64 , stride = 1 , padding = 1)
        self.layer1block2 = blocks(64 , 64 , stride = 1 , padding = 1)

        self.layer2block1 = blocks(64 , 128 , stride = 2 , padding = 1)
        self.layer2block2 = blocks(128 , 128 , stride = 1 , padding = 1) 

        self.layer3block1 = blocks(128 , 256 , stride = 2 , padding = 1)
        self.layer3block2 = blocks(256 , 256 , stride = 1 , padding = 1)

        self.layer4block1 = blocks(256 , 512 , stride = 2 , padding = 1)
        self.layer4block2 = blocks(512 , 512 , stride = 1 , padding = 1) 

        self.pooling_layer = nn.AdaptiveAvgPool2d((1 , 1))
        self.flatten = nn.Flatten()

        self.fc_layer = nn.Sequential(
            nn.Linear(512 , 1000),
            nn.ReLU(),
            nn.Linear(1000 , 2)
        )
    def forward(self , x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Relu_fnx(out)
        out = self.pooling(out)

        out = self.layer1block1(out)
        out = self.layer1block2(out)

        out = self.layer2block1(out)
        out = self.layer2block2(out)

        out = self.layer3block1(out)
        out = self.layer3block2(out)

        out = self.layer4block1(out)
        out = self.layer4block2(out) 

        out = self.pooling_layer(out)
        out = self.flatten(out)
        out = self.fc_layer(out)

        return out  


