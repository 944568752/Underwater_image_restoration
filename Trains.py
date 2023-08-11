

# Trans Cycle GAN ( Trains )


# Transformer in Cycle GAN


# Design by HanLin


import warnings
warnings.filterwarnings('ignore')


import os
import torch
import argparse
import itertools

from Datasets import Underwater_image_dataset

from Models import Generator
from Models import Discriminator

from Utils import LambdaLR
from Utils import Weight_initialization
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
parser.add_argument('--lrtestmodel',type=int,default=1)
# Test 1 : Use the same learning rate
parser.add_argument('--lr',type=float,default=0.0002)
# Test 2 : Use the different learning rates
parser.add_argument('--generatorlr',type=float,default=0.002)
parser.add_argument('--discriminatorlr',type=float,default=0.0001)

parser.add_argument('--size',type=int,default=256)

parser.add_argument('--inputchannel',type=int,default=3)
parser.add_argument('--outputchannel',type=int,default=3)

parser.add_argument('--numcpu',type=int,default=24)

args=parser.parse_args()


# Two generators
net_G_A2B=Generator(args.inputchannel,args.outputchannel)
net_G_B2A=Generator(args.inputchannel,args.outputchannel)

# Two discriminators
net_D_A=Discriminator(args.inputchannel)
net_D_B=Discriminator(args.inputchannel)


# Move to current device
net_G_A2B.to(device)
net_G_B2A.to(device)

net_D_A.to(device)
net_D_B.to(device)


# Initializa model parameters
net_G_A2B.apply(Weight_initialization)
net_G_B2A.apply(Weight_initialization)

net_D_A.apply(Weight_initialization)
net_D_B.apply(Weight_initialization)


# Loss function

# Discriminator loss
criterion_GAN = torch.nn.MSELoss()
# Cycle consistency loss
criterion_Cycle = torch.nn.L1Loss()
# Adversarial loss
criterion_Identity = torch.nn.L1Loss()


# Parameter optimizer
if args.lrtestmodel==1:
    optimizer_G=torch.optim.Adam(itertools.chain(net_G_A2B.parameters(),net_G_B2A.parameters()),lr=args.lr,betas=(0.5, 0.999))
    optimizer_D_A=torch.optim.Adam(net_D_A.parameters(),lr=args.lr,betas=(0.5, 0.999))
    optimizer_D_B=torch.optim.Adam(net_D_B.parameters(),lr=args.lr,betas=(0.5, 0.999))
elif args.lrtestmodel==2:
    optimizer_G = torch.optim.Adam(
            itertools.chain(net_G_A2B.parameters(),net_G_B2A.parameters()),
            lr=args.generatorlr,
            betas=(0.5, 0.999)
            )

    optimizer_D_A = torch.optim.Adam(
            net_D_A.parameters(),
            lr=args.discriminatorlr,
            betas=(0.5, 0.999)
            )

    optimizer_D_B = torch.optim.Adam(
            net_D_B.parameters(),
            lr=args.discriminatorlr,
            betas=(0.5, 0.999)
            )


# Adjust learning rate based on epoch
lr_scheduler_G=torch.optim.lr_scheduler.LambdaLR(optimizer_G,LambdaLR(args.totalepochs,args.startepoch,args.decayepoch).step)
lr_scheduler_D_A=torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,LambdaLR(args.totalepochs,args.startepoch,args.decayepoch).step)
lr_scheduler_D_B=torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,LambdaLR(args.totalepochs,args.startepoch,args.decayepoch).step)


Start_epoch=args.startepoch


# Load network parameters
if args.loadparameter == True:
    if os.path.exists(args.parameterpath):
        Parameter_backup = torch.load(args.parameterpath)

        Start_epoch=Parameter_backup['epoch']

        net_G_A2B.load_state_dict(Parameter_backup['net_G_A2B'])
        net_G_B2A.load_state_dict(Parameter_backup['net_G_B2A'])

        net_D_A.load_state_dict(Parameter_backup['net_D_A'])
        net_D_B.load_state_dict(Parameter_backup['net_D_B'])

        optimizer_G.load_state_dict(Parameter_backup['optimizer_G'])
        optimizer_D_A.load_state_dict(Parameter_backup['optimizer_D_A'])
        optimizer_D_B.load_state_dict(Parameter_backup['optimizer_D_B'])

        lr_scheduler_G.load_state_dict(Parameter_backup['lr_scheduler_G'])
        lr_scheduler_D_A.load_state_dict(Parameter_backup['lr_scheduler_D_A'])
        lr_scheduler_D_A.load_state_dict(Parameter_backup['lr_scheduler_D_B'])


# Load training data
dataloader = DataLoader(
    Underwater_image_dataset(args.datapath,List_transforms_operation(256),mode='train'),
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=args.numcpu
)


target_real=torch.ones(args.batchsize,device=device,requires_grad=False)
target_fake=torch.zeros(args.batchsize,device=device,requires_grad=False)
# target_real.to(device)
# target_fake.to(device)


if __name__=='__main__':

    net_G_A2B.train()
    net_G_B2A.train()
    net_D_A.train()
    net_D_B.train()

    for epoch in range(Start_epoch,args.totalepochs):

        Total_loss_G=0
        Total_loss_D=0

        Total_loss_GAN=0
        Total_loss_Cycle=0
        Total_loss_Identity=0

        for i,batch in enumerate(dataloader):

            real_A=batch['A'].to(device)
            real_B=batch['B'].to(device)

            # Generator A2B & B2A
            optimizer_G.zero_grad()

            same_B=net_G_A2B(real_B)
            loss_Identity_B=criterion_Identity(same_B,real_B)

            same_A=net_G_B2A(real_A)
            loss_Identity_A=criterion_Identity(same_A,real_A)


            fake_B=net_G_A2B(real_A)
            similarity_B=net_D_B(fake_B)
            loss_GAN_A2B=criterion_GAN(similarity_B,target_real)

            fake_A=net_G_B2A(real_B)
            similarity_A=net_D_A(fake_A)
            loss_GAN_B2A=criterion_GAN(similarity_A,target_real)


            recover_A=net_G_B2A(fake_B)
            loss_Cycle_ABA=criterion_Cycle(recover_A,real_A)

            recover_B=net_G_A2B(fake_A)
            loss_Cycle_BAB=criterion_Cycle(recover_B,real_B)


            # Initial : [5,1,10]
            loss_G=5*(loss_Identity_A+loss_Identity_B)+1*(loss_GAN_A2B+loss_GAN_B2A)+10*(loss_Cycle_ABA+loss_Cycle_BAB)

            loss_G.backward()
            optimizer_G.step()


            # Discirminator A
            optimizer_D_A.zero_grad()

            similarity_A=net_D_A(real_A)
            loss_D_A_real=criterion_GAN(similarity_A,target_real)

            # detach : Prevent Pytorch from releasing the compute graph
            similarity_A=net_D_A(fake_A.detach())
            loss_D_A_fake=criterion_GAN(similarity_A,target_fake)

            loss_D_A=0.5*loss_D_A_real+0.5*loss_D_A_fake


            loss_D_A.backward()
            optimizer_D_A.step()


            # Discriminator B
            optimizer_D_B.zero_grad()

            similarity_B=net_D_B(real_B)
            loss_D_B_real=criterion_GAN(similarity_B,target_real)

            # detach : Prevent Pytorch from releasing the compute graph
            similarity_B=net_D_B(fake_B.detach())
            loss_D_B_fake=criterion_GAN(similarity_B,target_fake)

            loss_D_B=0.5*loss_D_B_real+0.5*loss_D_B_fake


            loss_D_B.backward()
            optimizer_D_B.step()


            Total_loss_G = Total_loss_G+loss_G
            Total_loss_D = Total_loss_D+(loss_D_A+loss_D_B)

            Total_loss_GAN = Total_loss_GAN+(loss_GAN_A2B+loss_GAN_B2A)
            Total_loss_Cycle = Total_loss_Cycle+(loss_Cycle_ABA+loss_Cycle_BAB)
            Total_loss_Identity = Total_loss_Identity+(loss_Identity_A+loss_Identity_B)


        print(f'Epoch : [{epoch+1} / {args.totalepochs}] || Loss_G : {Total_loss_G/(i+1):.7f} | Loss_D : {Total_loss_D/(i+1):.7f}')
        print(f'Loss_GAN : {Total_loss_GAN/(i+1):.7f} | Loss_Cycle : {Total_loss_Cycle/(i+1):.7f} | Loss_Identity : {Total_loss_Identity/(i+1):.7f}')
        print('\n')


        if epoch%9==0:

            # Save network parameters
            Parameter_backup={
                'epoch':epoch,

                'net_G_A2B':net_G_A2B.state_dict(),
                'net_G_B2A':net_G_B2A.state_dict(),
                'net_D_A':net_D_A.state_dict(),
                'net_D_B':net_D_B.state_dict(),

                'optimizer_G':optimizer_G.state_dict(),
                'optimizer_D_A':optimizer_D_A.state_dict(),
                'optimizer_D_B':optimizer_D_B.state_dict(),

                'lr_scheduler_G':lr_scheduler_G.state_dict(),
                'lr_scheduler_D_A':lr_scheduler_D_A.state_dict(),
                'lr_scheduler_D_B':lr_scheduler_D_B.state_dict()
            }

            torch.save(Parameter_backup,args.parameterpath)

















