import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10


dataset = CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

dataLoader = DataLoader(dataset,64)

class myModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = torch.nn.Conv2d(3,6,3)

    def forward(self,x):
        x = self.conv(x)
        return x

writer = SummaryWriter('logs')

model = myModel()

step = 0
for data in dataLoader:
    imgs,targets = data
    writer.add_images('input',imgs,step)
    output = model(imgs)
    output = torch.reshape(output,(-1,3,30,30))
    print(output.shape)
    writer.add_images('output',output,step)
    step +=1


