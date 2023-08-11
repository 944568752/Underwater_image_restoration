

# Trans Cycle GAN ( Utils )


# Transformer in Cycle GAN


# Design by HanLin


import os
import argparse
import torch

from Datasets import Underwater_image_dataset

from Models import Generator

from torchvision.utils import save_image

from Utils import List_transforms_operation

from torch.utils.data import DataLoader


if torch.cuda.is_available():
    device=torch.device('cuda:0')
    print('The current device is GPU !',end='\n\n')
else:
    device=torch.device('cpu')
    print('The current device is CPU !',end='\n\n')


parser=argparse.ArgumentParser()

parser.add_argument('--datapath',type=str,default=r'./Underwater_image_dataset_for_ViT_Cycle_GAN')

parser.add_argument('--testresultpath',type=str,default=r'./Test_result')

parser.add_argument('--loadparameter',type=bool,default=True)
parser.add_argument('--parameterpath',type=str,default=r'./Parameter_backup/Trans_Cycle_GAN_Parameters.pth')

parser.add_argument('--startepoch',type=int,default=0)
parser.add_argument('--decayepoch',type=int,default=470)

parser.add_argument('--totalepochs',type=int,default=980)
parser.add_argument('--batchsize',type=int,default=1)

# learning rate test : 0.1/0.01/0.001/0.0001
parser.add_argument('--lrtestmodel',type=int,default=2)
# Test 1 : Use the same learning rate
parser.add_argument('--lr',type=float,default=0.002)
# Test 2 : Use the different learning rates
parser.add_argument('--generatorlr',type=float,default=0.005)
parser.add_argument('--discriminatorlr',type=float,default=0.001)

parser.add_argument('--size',type=int,default=256)

parser.add_argument('--inputchannel',type=int,default=3)
parser.add_argument('--outputchannel',type=int,default=3)

parser.add_argument('--numcpu',type=int,default=24)

args=parser.parse_args()


# Two generators
net_G_A2B=Generator(args.inputchannel,args.outputchannel)
net_G_B2A=Generator(args.inputchannel,args.outputchannel)


# Move to current device
net_G_A2B.to(device)
net_G_B2A.to(device)


# Load network parameters
if args.loadparameter == True:
    Parameter_backup = torch.load(args.parameterpath)

    net_G_A2B.load_state_dict(Parameter_backup['net_G_A2B'])
    net_G_B2A.load_state_dict(Parameter_backup['net_G_B2A'])


# Set model's test mode
net_G_A2B.eval()
net_G_B2A.eval()


# Load testing data
dataloader = DataLoader(
    Underwater_image_dataset(args.datapath,List_transforms_operation(256),mode='test'),
    batch_size=args.batchsize,
    shuffle=False,
    num_workers=args.numcpu
)


# Check the test result output path
test_result_path_A = os.path.join(args.testresultpath,'A')
test_result_path_B = os.path.join(args.testresultpath,'B')

if not os.path.exists(test_result_path_A):
    os.makedirs(test_result_path_A)
if not os.path.exists(test_result_path_B):
    os.makedirs(test_result_path_B)


with torch.no_grad():
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        fake_B = net_G_B2A(real_A)
        fake_A = net_G_A2B(real_B)


        save_image(real_A,os.path.join(test_result_path_A,f'real_{i+1}.png'))
        save_image(fake_B,os.path.join(test_result_path_A,f'fake_{i+1}.png'))

        save_image(real_B,os.path.join(test_result_path_B,f'real_{i+1}.png'))
        save_image(fake_A,os.path.join(test_result_path_B,f'fake_{i+1}.png'))

        print(f'\r Generated images {i+1} of {len(dataloader)} !')











