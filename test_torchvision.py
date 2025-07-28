import torchvision

train = torchvision.datasets.CIFAR10(root='./dataset',train=True,download=True)
test = torchvision.datasets.CIFAR10(root='./dataset',train=False,download=True)

print(train[0])
print('hello')