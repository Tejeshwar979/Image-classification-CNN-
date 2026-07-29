from utils.load import *  
from utils.model import *  
from utils.ResNet import * 

model = ResNet18() 

try:
    model.load_state_dict(torch.load("ResNet_epoch20_model.pth"))

    ct = 0 
    size = 0 
    for img , lable in test_dataloader : 
       outputs = model(img)
       _ , predicted = torch.max(outputs , 1)
       size += lable.size(0)
       ct += (predicted == lable).sum().item()
    print((ct/size) * 100)

except Exception as e : 
    print(e)


