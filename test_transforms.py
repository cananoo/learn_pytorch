from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


path = r"C:\Users\ASUS\Desktop\dataset\train\ants\67270775_e9fdf77e9d.jpg"

img = Image.open(path)

tensorTool = transforms.ToTensor()

tensor_img = tensorTool(img)

writer = SummaryWriter('logs')

print(tensor_img[0,0,0])
normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
nor_img = normalize(tensor_img)
print(nor_img[0,0,0])
writer.add_image('normalization',nor_img)



