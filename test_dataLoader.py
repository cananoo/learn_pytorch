import torch.utils.data
import torchvision
from torch.utils.tensorboard import SummaryWriter



test_data = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor())

loader = torch.utils.data.DataLoader(test_data,4,False,num_workers=0,drop_last=False)

step = 0
writer = SummaryWriter('logs')
for data in loader:
    imgs,targets = data
    writer.add_images('loader',imgs,step)
    step+=1
    print(targets)

writer.close()