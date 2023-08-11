

# Trans Cycle GAN ( Datasets )


# Transformer in Cycle GAN


# Design by HanLin


import warnings
warnings.filterwarnings('ignore')


import os
import glob
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


# Trans Cycle GAN's underwater image data set

# Input :
# image_path
# transforms_operation
# mode

class Underwater_image_dataset(Dataset):
    def __init__(self,image_path,transforms_operation,mode='train'):
        self.transform = transforms.Compose(transforms_operation)

        # mode:
        # train
        # test

        # Get all image paths
        self.image_path=os.path.join(image_path,mode)
        # Underwater image
        self.image_paths_A = glob.glob(os.path.join(self.image_path,'A','*.*'))
        # Normal image
        self.image_paths_B = glob.glob(os.path.join(self.image_path,'B','*.*'))

        # Shuffle paths
        random.shuffle(self.image_paths_A)
        random.shuffle(self.image_paths_B)

    def __getitem__(self,item):
        # Method 1
        # image_A=self.transform(Image.open(self.image_paths_A(item)))
        # image_B=self.transform(Image.open(self.image_paths_B(item)))

        # Method 2
        image_A=self.transform(Image.open(self.image_paths_A[random.randint(0, len(self.image_paths_A) - 1)]).convert('RGB'))
        image_B=self.transform(Image.open(self.image_paths_B[random.randint(0, len(self.image_paths_B) - 1)]).convert('RGB'))

        return {'A':image_A,'B':image_B}
        # Output :
        # A : image_A
        # B : image_B

    def __len__(self):
        return max(len(self.image_paths_A),len(self.image_paths_B))










