import torchvision.transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

import torch
import time


writer = SummaryWriter('logs')

train_data = CIFAR10('./dataset',True,torchvision.transforms.ToTensor(),download=True)
test_data = CIFAR10('./dataset',False,torchvision.transforms.ToTensor(),download=True)

print(len(train_data))
print(len(test_data))


train_loader = DataLoader(train_data,64)
test_loader = DataLoader(test_data,64)

train_steps = 0
test_steps = 0
epoch = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
model = Model()
model.to(device)

cross_loss = CrossEntropyLoss()
cross_loss.to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


start_time = time.time()
for i in range(epoch):
    print("-------第{}轮训练------".format(i+1))
    for data in train_loader:
        imgs,targets = data

        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        end_time = time.time()
        loss = cross_loss(outputs,targets)
        if (train_steps%100 == 0):
            print("第{}批，loss为{},训练时间为{}".format(train_steps+1,loss.item(),end_time-start_time))
            writer.add_scalar('train_loss',loss,train_steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_steps += 1

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = cross_loss(outputs, targets)

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
            test_steps +=1
            total_test_loss += loss
    writer.add_scalar('test_loss',total_test_loss,i+1)
    print("整体的测试loss值为{}".format(total_test_loss))
    print("整体的测试准确率为{}".format(total_accuracy/len(test_data)))
torch.save(model,'model.pth')
print('模型已保存')
writer.close()