from PIL import  Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


path = r"C:\Users\ASUS\Desktop\dataset\train\ants\67270775_e9fdf77e9d.jpg"
img = Image.open(path)

transforms_resize = transforms.Resize((100,100))
transforms_toTensor = transforms.ToTensor()
transforms_randomCrop = transforms.RandomCrop(20)
compose = transforms.Compose([transforms_resize,transforms_randomCrop,transforms_toTensor])

tensor_img = compose(img)

writer = SummaryWriter('logs')
writer.add_image("tensor_img",tensor_img,1)


