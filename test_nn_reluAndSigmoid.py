import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10


dataset = CIFAR10('./dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())
dataLoader = DataLoader(dataset,64)

input = torch.tensor([[1,-0.5],[-0.3,2]])

input = torch.reshape(input,(1,2,2))

class UnlineModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        output = self.sigmoid(x)
        return output

sigmoid = UnlineModel()
writer = SummaryWriter('logs')
step = 0
for data in dataLoader:
    imgs,targest = data
    writer.add_images('input',imgs,step)
    output = sigmoid(imgs)
    writer.add_images('output', output, step)
    step +=1

