
import torch 
import torch.nn as nn  

class fo(nn.Module) : 
     def __init__(self) : 
        super().__init__()
        self.fo_model = nn.Sequential(
            nn.Conv2d(3 , 64 , 7 , stride = 2 , padding = 1) 
        )




