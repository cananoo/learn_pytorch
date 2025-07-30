import torchvision.transforms
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

from model import *
import torch



writer = SummaryWriter('logs')

train_data = CIFAR10('./dataset',True,torchvision.transforms.ToTensor(),download=True)
test_data = CIFAR10('./dataset',False,torchvision.transforms.ToTensor(),download=True)

print(len(train_data))
print(len(test_data))

train_loader = DataLoader(train_data,64)
test_loader = DataLoader(test_data,64)

train_steps = 0
test_steps = 0
epoch = 10

model = Model()

cross_loss = CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


for i in range(epoch):
    print("-------第{}轮训练------".format(i+1))
    for data in train_loader:
        imgs,targets = data
        outputs = model(imgs)
        loss = cross_loss(outputs,targets)
        if (train_steps%100 == 0):
            print("第{}批，loss为{}".format(train_steps+1,loss.item()))
            writer.add_scalar('train_loss',loss,train_steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_steps += 1
    torch.save(model,'model_{}_epoch.pth')

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs,targets = data
            outputs = model(imgs)
            loss = cross_loss(outputs, targets)
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
            test_steps +=1
            total_test_loss += loss
    writer.add_scalar('test_loss',total_test_loss,i+1)
    print("整体的测试loss值为{}".format(total_test_loss))
    print("整体的测试准确率为{}".format(total_accuracy/len(test_data)))
    print('模型已保存')

writer.close()