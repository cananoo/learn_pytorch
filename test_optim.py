import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


dataset = CIFAR10('./dataset',train=False,download=True,transform=torchvision.transforms.ToTensor())

dataLoader = DataLoader(dataset,1)

class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, padding=2, stride=1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, padding=2, stride=1),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, padding=2, stride=1),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 64),
            torch.nn.Linear(64, 10)
        )

    def forward(self,x):

        x = self.model1(x)
        return  x

mod = Model()


# crossEntropyLoss
input = torch.tensor([0.1,0.2,0.3])
target = torch.tensor([1])
input = torch.reshape(input,(1,3))
cross = torch.nn.CrossEntropyLoss()
print(cross(input,target))


optim = torch.optim.SGD(mod.parameters(),lr=0.01)


for epoch in range(10):
    total_loss = 0.0
    for data in dataLoader:
        imgs,targets = data
        output = mod(imgs)
        loss = cross(output,targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss
    print(total_loss)






