import  torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


dataset = CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataLoader = DataLoader(dataset,64)

class LinerModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.liner = torch.nn.Linear(196608,10) #from 196608 to 10

    def forward(self,input):
        output = self.liner(input)
        return output

liner = LinerModel()
for data in dataLoader:
    imgs,targets = data
    output = torch.flatten(imgs)
    print(output.shape)
    output = liner(output)
    print(output.shape)