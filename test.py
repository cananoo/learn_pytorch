import torchvision.transforms
from PIL import Image
import torch

img_path = './test_imgs/test.png'

image = Image.open(img_path)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                torchvision.transforms.ToTensor()])
input = transform(image)

input = torch.reshape(input,[1,3,32,32])

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
model = torch.load('model.pth',map_location=torch.device('cpu'),weights_only=False)
print(model)
model.eval()
with torch.no_grad():
    output = model(input)
print(output)
print('识别成功'if output.argmax(1).item() == 1 else '识别错误')
