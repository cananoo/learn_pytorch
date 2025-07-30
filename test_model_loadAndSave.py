import torch
from torchvision.models import vgg16

vgg = vgg16(pretrained=False)

#save_method_1
torch.save(vgg,'model_saved1.pth')

#save_method_2
torch.save(vgg.state_dict(),'model_saved2.pth')


#load_method_1
model = torch.load('model_saved1.pth',weights_only=False)
print(model)

#load_method_2
vgg2 = vgg16(pretrained=False)
vgg2.load_state_dict(torch.load('model_saved2.pth'))
print(vgg2)