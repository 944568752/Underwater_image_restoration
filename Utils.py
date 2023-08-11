

# Trans Cycle GAN ( Utils )


# Transformer in Cycle GAN


# Design by HanLin


import warnings
warnings.filterwarnings('ignore')


import torch


from PIL import Image
from torchvision.transforms import transforms


# List the operations that need to be performed on the dataset image
def List_transforms_operation(image_size,zoom_facter = 1.25):

    image_size = int(image_size)

    transforms_operations = []

    # Aspect ratio change
    # transforms_operations.append(transforms.Resize([image_size,image_size],Image.BICUBIC))

    # Aspect ratio unchange
    transforms_operations.append(transforms.Resize(int(image_size * zoom_facter), Image.BICUBIC))

    transforms_operations.append(transforms.RandomCrop(image_size))

    transforms_operations.append(transforms.RandomHorizontalFlip(p = 0.5))
    transforms_operations.append(transforms.RandomVerticalFlip(p = 0.5))

    transforms_operations.append(transforms.ToTensor())

    # transforms_operations.append(transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)))

    return transforms_operations


# Network initialization

# Use .apply(Weight_initialization) function call, net class and its subclasses will be called

def Weight_initialization(model):
    # __class__ : View the class of the object
    # __name__ : Get the class name
    classname=model.__class__.__name__

    # Initialization of the convolutional layer
    if classname.find('Conv') != -1:
        # 对权重进行正态分布初始化
        # 目标权重
        # 平均值
        # 标准差
        torch.nn.init.normal(model.weight.data,0.0,0.02)

    # Initialization of the batchnormalization layer
    elif classname.find('BatchNorm') != -1:
        # 对权重进行正态分布初始化
        # 目标权重
        # 平均值
        # 标准差
        torch.nn.init.normal(model.weight.data,1.0,0.02)
        # 对偏差进行常量初始化
        # 目标权重
        # 常量
        torch.nn.init.constant(model.bias.data,0.0)


# Calculate the learning rate update
class LambdaLR():
    def __init__(self,epochs,offset,decay_start_epoch):

        assert epochs > decay_start_epoch,'Decay can only be apply at the end of training session !'

        self.epochs = epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self,epoch):
        return 1.0 - max(0,epoch+self.offset-self.decay_start_epoch)/(self.epochs-self.decay_start_epoch)










