from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image


path = r"C:\Users\ASUS\Desktop\dataset\train\ants\67270775_e9fdf77e9d.jpg"

img = Image.open(path)

img_array = np.array(img)

writer = SummaryWriter('logs')

writer.add_image("ants",img_array,1,dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=x",2*i,i)

writer.close()