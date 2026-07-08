import torch 
import torch.nn as nn 

import torch.optim as optim 

from utils.model import *
from utils.load import * 

model  = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN.parameters() , lr = 0.001)

device = torch.device("cuda")

epochs = 10 

for epoch in range(epochs):
    loss_per_epoch = 0.0 
    for img , labels in traindata:
        
        optimizer.zero_grad() 

        output = model(img)

        loss = criterion(output , labels)

        loss.backward()

        optimizer.step()

        loss_per_epoch += loss 
    print(f" {epoch + 1} / {epochs}  loss = {loss_per_epoch / len(traindata)} ")

try : 
    torch.save(
        model.state_dict() , 
        f"train.pth" , 
    )
except Exception as e : 
    print(e)


