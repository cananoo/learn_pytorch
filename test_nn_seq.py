import torch
from torch.utils.tensorboard import SummaryWriter



class Model(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.conv1 = torch.nn.Conv2d(3,32,5,padding=2,stride=1)
        # self.pool1 = torch.nn.MaxPool2d(2)
        # self.conv2 = torch.nn.Conv2d(32, 32, 5, padding=2, stride=1)
        # self.pool2 = torch.nn.MaxPool2d(2)
        # self.conv3 = torch.nn.Conv2d(32, 64, 5, padding=2, stride=1)
        # self.pool3 = torch.nn.MaxPool2d(2)
        # self.flatten = torch.nn.Flatten()
        # self.liner1 = torch.nn.Linear(64*4*4,64)
        # self.liner2 = torch.nn.Linear(64,10)

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
        # x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        # x = self.flatten(x)
        # x = self.liner1(x)
        # x = self.liner2(x)
        x = self.model1(x)
        return  x

mod = Model()
input = torch.ones([64,3,32,32])
output = mod(input)
print(output.shape)

writer = SummaryWriter('logs')
writer.add_graph(mod,input)
writer.close()