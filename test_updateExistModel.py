import torch
from torchvision.models import vgg16

vgg_false = vgg16(pretrained=False)
vgg_true = vgg16(pretrained=True)

print(vgg_false)
vgg_false.classifier[6] = torch.nn.Linear(4096,10)
print(vgg_false)

print(vgg_true)
vgg_true.classifier.add_module('add_liner',torch.nn.Linear(1000,10))
print(vgg_true)