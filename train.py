import torch 
import torch.nn as nn 

import torch.optim as optim 

from utils.model import *
from utils.load import * 

model  = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr = 0.001)

device = torch.device("cuda")

epochs = 100

try : 
# training the model  
    for epoch in range(epochs):
        loss_per_epoch = 0.0 
        for img , labels in train_dataloader:
            img = img.to(device)
            labels = labels.to(device)
            optimizer.zero_grad() 

            output = model(img)

            loss = criterion(output , labels)

            loss.backward()

            optimizer.step()

            loss_per_epoch += loss 
        print(f" {epoch + 1} / {epochs}  loss = {loss_per_epoch / len(train_dataloader)} ")
# saving the model  and exception in thde model saving 
    try : 
        torch.save(
            model.state_dict() , 
            f"CNN_base_model.pth" , 
        )
    except Exception as e : 
        print(e)

# exception in the training  
except Exception as e : 
    print(e)


