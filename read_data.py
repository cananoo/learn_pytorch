from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    def __init__(self,root_path,label_path):
        self.root_path = root_path
        self.label_path = label_path
        self.path = os.path.join(root_path,label_path)
        self.path_list = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.path_list[idx]
        img = Image.open(os.path.join(self.path,img_name))
        return img,self.label_path


    def __len__(self):
        return len(self.path_list)

antsDataset = MyData(r'C:\Users\ASUS\Desktop\dataset\train','ants')
img,label = antsDataset.__getitem__(1)
img.show()
print(label)