import torch
import torch.nn as nn 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import datetime
from src.model.modelTanoGan import *
import torch.nn.init as init
from src.utils.util import *
from collections import OrderedDict
from src.utils.tf_dtw import SoftDTW

class TanoGan:
    
    def __init__(self,device=None,opt_trn=None,windows_length=60,n_features=1,embedding_dim=16):
        self.embedding_dim=embedding_dim
        self.device=device
        self.lr=opt_trn.lr
        self.windows_length=windows_length
        self.in_dim=n_features
        self.epochs=opt_trn.epochs
        self.criterion = torch.nn.MSELoss()
        self.netD = LSTMDiscriminator(in_dim=self.in_dim,device=self.device)
        self.netD=nn.DataParallel(self.netD)
        self.netD=self.netD.to(self.device)
        self.netG = LSTMGenerator(in_dim=self.in_dim, out_dim=self.in_dim,device=self.device)
        self.netG=nn.DataParallel(self.netG)
        self.netG=self.netG.to(self.device)
        
        self.optimizerD = optim.Adam(self.netD.parameters() , lr=self.lr)
        self.optimizerG=  optim.Adam(self.netG.parameters(),  lr=self.lr)


    def load_model(self, state_dict, model):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict)
        return model
    
    def intilialize_D(self,model):
        self.netD = model
        
    def intilialize_G(self,model):
        self.netG = model
        
    def train_TanoGAN(self,sequences):
        real_label=1
        fake_label=0
        for epoch in range(0,self.epochs):
            for x in sequences:
                #Train with real data
                self.optimizerD.zero_grad()
                real=x
                real=real.float().to(self.device)
                batch_size,seq_len=real.size(0),real.size(1)
                label=torch.full((batch_size,seq_len,1),real_label)

                output,_ = self.netD.forward(real.to(self.device))
                errD_real=self.criterion(output.to(self.device),label.float().to(self.device))
               
                D_x=output.mean().item()

                #Train with fake data
                noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,self.in_dim),mean=0,std=0.1)).to(self.device)
                fake,_ = self.netG.forward(noise)
                output,_=self.netD.forward(fake.detach().to(self.device)) # detach causes gradient is no longer being computed or stored to save memeory
                label.fill_(fake_label)
                errD_fake=self.criterion(output.to(self.device),label.float().to(self.device))
               
                D_G_z1 = output.mean().item()

                errD = errD_real+errD_fake
                errD.backward()
                self.optimizerD.step()

                 ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.optimizerG.zero_grad()
                noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,self.in_dim),mean=0,std=0.1)).to(self.device)
                fake,_ = self.netG.forward(noise)
                label.fill_(real_label) 
                output,_ = self.netD.forward(fake.to(self.device))
                errG = self.criterion(output.to(self.device), label.float().to(self.device))
                errG.backward()
                self.optimizerG.step()
                D_G_z2 = output.mean().item()

            print("Completed epoch "+ str(epoch))
        return self.netD,self.netG

    def Anomaly_score(self,x, G_z, Lambda=0.1):
        residual_loss = torch.sum(torch.abs(x-G_z)) # Residual Loss

        # x_feature is a rich intermediate feature representation for real data x
        output, x_feature = self.netG(x) 
        # G_z_feature is a rich intermediate feature representation for fake data G(z)
        output, G_z_feature = self.netG(G_z) 

        discrimination_loss = torch.sum(torch.abs(x_feature-G_z_feature)) # Discrimination loss

        total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss
        return total_loss

    def predict_loss(self,sequences,batch_size=1):
        loss_list = []
        #y_list = []
        for x in sequences:

            z = Variable(init.normal(torch.zeros(batch_size,
                                             self.windows_length, 
                                             self.in_dim),mean=0,std=0.1),requires_grad=True)
            #z = x
            z_optimizer = torch.optim.Adam([z],lr=1e-2)

            loss = None
            for j in range(50): # set your interation range
                gen_fake,_ = self.netG(z)
                loss = self.Anomaly_score(Variable(x).float().to(self.device), gen_fake.float().to(self.device))
                loss.backward()
                z_optimizer.step()

            loss_list.append(loss) # Store the loss from the final iteration
            #y_list.append(y) # Store the corresponding anomaly label
            #print('~~~~~~~~loss={},  y={} ~~~~~~~~~~'.format(loss, y))
            #break
        return loss_list