import torch
import torch.nn as nn 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import datetime
from src.model.modelGranoGan import Encoder,Decoder,Critic
from src.utils.losses import Wasserstein
import torch.nn.init as init
from src.utils.util import *
from collections import OrderedDict
from src.utils.tf_dtw import SoftDTW

class GranoGAN:
    def __init__(self,device=None,opt_trn=None,windows_length=60,n_features=1,in_dim=1):
        self.in_dim=in_dim
        self.device=device
        self.lr=opt_trn.lr
        self.windows_length=windows_length
        self.in_dim=n_features
        self.epochs=opt_trn.epochs
        self.mse_loss = torch.nn.MSELoss()
        self.mean=opt_trn.mean
        self.std=opt_trn.std
  
        
        #initialize criticx and decoder
        self.criticX = Critic(in_dim=self.in_dim,device=self.device)
        self.criticX=nn.DataParallel(self.criticX)
        self.criticX=self.criticX.to(self.device)
        self.optimizerCriticX = optim.Adam(self.criticX.parameters() , lr=self.lr)
        self.decoder=Decoder(in_dim=self.in_dim, out_dim=self.in_dim,device=self.device)
        self.decoder=nn.DataParallel(self.decoder)
        self.decoder=self.decoder.to(self.device)
        self.optimizerDecoder = optim.Adam(self.decoder.parameters(), lr=self.lr)
        
        
        #Initialize criticz and encoder
        self.criticZ = Critic(in_dim=in_dim,device=device)
        self.criticZ=nn.DataParallel(self.criticZ)
        self.criticZ=self.criticZ.to(self.device)
        self.optimizerCriticZ = optim.Adam(self.criticZ.parameters() , lr=self.lr)
        self.encoder=Encoder(n_features=self.in_dim, embedding_dim=self.in_dim,device=self.device)
        self.encoder=nn.DataParallel(self.encoder)
        self.encoder=self.encoder.to(self.device)
        self.optimizerEncoder = optim.Adam(self.encoder.parameters(), lr=self.lr)
        
        #define loss function
        self.loss_fn=Wasserstein()
    
    def initialize_models(self,criticX,criticZ,decoder,encoder):
        self.criticX=criticX
        self.criticZ=criticZ
        self.decoder=decoder
        self.encoder=encoder
        
    def initialize_criticX(self,criticX):
        self.criticX=criticX
    
    def initialize_criticZ(self,criticZ):
        self.criticZ=criticZ
        
    def initialize_decoder(self,decoder):
        self.decoder=decoder
        
    def initialize_encoder(self,encoder):
        self.encoder=encoder
        
        
    def load_model(self,state_dict,model):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k]=v
        model.load_state_dict(new_state_dict)
        return model
    
    def consistency_normalize_gradient(model, x, **kwargs):
        """
                         f
        f_hat = --------------------
                || grad_f || + | f |
        """
        x.requires_grad_(True)
        f = model(x, **kwargs)
        grad = torch.autograd.grad(
            f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
        grad_norm = torch.norm(torch.flatten(grad, start_dim=2), p=2, dim=2)
        grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])

        f_hat = (f / (grad_norm + torch.abs(f)))
        return f_hat

    def train_criticX(self,sequences):
        
        #Training CriticX
        history_criticX = dict(train=[], val=[])
        for epoch in range(self.epochs):
            train_losses = []
            for x in sequences:
                temp_train_losses = []
                batch_size, seq_len =x.shape[0],x.shape[1] 
                x=x.float().to(self.device)
                noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,self.in_dim),mean=self.mean,std=self.std))
                noise=noise.to(self.device)
                self.optimizerCriticX.zero_grad()
                for k in range(5):
                    with torch.no_grad():
                        outputOfDecoder,_ = self.decoder.forward(noise)
                        #outputOfDecoder=outputOfDecoder.detach()
                        outputOfDecoder=outputOfDecoder.to(self.device)
                    real_fake = torch.cat([x, outputOfDecoder], dim=0)
                    with torch.backends.cudnn.flags(enabled=False):
                        pred = normalize_gradient(self.criticX, real_fake)
                        pred_real, pred_fake = torch.split(pred, [x.shape[0]*x.shape[1], outputOfDecoder.shape[0]*outputOfDecoder.shape[1]])
                        loss, loss_real, loss_fake = self.loss_fn(pred_real, pred_fake)
                        loss.backward()    
                        self.optimizerCriticX.step()
                    temp_train_losses.append(loss.item())
                train_losses.append(np.mean(temp_train_losses))
            train_loss = np.mean(train_losses)
            history_criticX['train'].append(train_loss)    
            print(f'Epoch {epoch}: train loss {train_loss}')
        return self.criticX
        
    def train_criticZ(self,sequences):
        #Training criticZ
        history_criticZ = dict(train=[], val=[])
        for epoch in range(self.epochs):
            train_losses = []
            for x in sequences:
                temp_train_losses = []
                batch_size, seq_len =x.shape[0],x.shape[1] 
                x=x.float().to(self.device)
                noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,self.in_dim),mean=self.mean,std=self.std))
                noise=noise.to(self.device)
                self.optimizerCriticZ.zero_grad()
                for _ in range(5):
                    with torch.no_grad():
                        outputOfEncoder,_ = self.encoder.forward(x)
                        outputOfEncoder=outputOfEncoder.detach()
                        outputOfEncoder=outputOfEncoder.to(self.device)
                    real_fake = torch.cat([noise, outputOfEncoder], dim=0)
                    with torch.backends.cudnn.flags(enabled=False):
                        pred = normalize_gradient(self.criticZ, real_fake)
                        pred_real, pred_fake = torch.split(pred, [x.shape[0]*x.shape[1], outputOfEncoder.shape[0]*outputOfEncoder.shape[1]])
                        loss, loss_real, loss_fake = self.loss_fn(pred_real, pred_fake)
                        loss.backward()    
                        self.optimizerCriticZ.step()
                    temp_train_losses.append(loss.item())
                train_losses.append(np.mean(temp_train_losses))
            train_loss = np.mean(train_losses)
            history_criticZ['train'].append(train_loss)    
            print(f'Epoch {epoch}: train loss {train_loss}')
        return self.criticZ
        
    def train_enc_dec(self,sequences):
        #Training encoder and decoder simlutaneously
        history_enc_dec = dict(train=[], val=[])
        for epoch in range(self.epochs):
        #for epoch in range(2):
            train_losses = []
            for x in sequences:
                batch_size, seq_len =x.shape[0],x.shape[1]
                x=x.float().to(self.device)
                self.optimizerDecoder.zero_grad()
                noise = Variable(init.normal(torch.Tensor(batch_size,seq_len,self.in_dim),mean=self.mean,std=self.std))
                noise=noise.to(self.device)
                outputOfDecoder,_ = self.decoder.forward(noise)
                outputOfDecoder=outputOfDecoder.detach()
                outputOfDecoder=outputOfDecoder.to(self.device)
                with torch.backends.cudnn.flags(enabled=False):
                    pred1 = normalize_gradient(self.criticX, outputOfDecoder)
                    enc_z,_=self.encoder.forward(x)
                    dec_x,_=self.decoder.forward(enc_z)

                mse1=self.mse_loss(x,dec_x)
                loss1=self.loss_fn(pred1)+mse1
                self.optimizerEncoder.zero_grad()

                outputOfEncoder,_ = self.encoder.forward(x)
                outputOfEncoder=outputOfEncoder.detach()
                outputOfEncoder=outputOfEncoder.to(self.device)
                with torch.backends.cudnn.flags(enabled=False):
                    pred2 = normalize_gradient(self.criticZ, outputOfEncoder)
                    dec_x,_=self.decoder.forward(noise)
                    enc_z,_=self.encoder.forward(dec_x)

                mse2=self.mse_loss(noise,enc_z)
                loss2=self.loss_fn(pred2)+mse2
                err=loss1+loss2
                err.backward()
                self.optimizerDecoder.step()
                self.optimizerEncoder.step()

                train_losses.append(err.item())
            train_loss = np.mean(train_losses)
            history_enc_dec['train'].append(train_loss)
            print(f'Epoch {epoch}: train loss {train_loss}')
        return self.encoder,self.decoder
        
    def predict_loss(self,sequences): 
        criterion_dtw = SoftDTW(gamma=1.0, normalize=True) # just like nn.MSELoss()
        losses = []
        criterion =  nn.L1Loss(reduction='sum')
        for x in sequences:
            x=x.float().to(self.device)
            enc_z,_=self.encoder.forward(x)
            enc_z=enc_z.to(self.device)
            dec_x,_=self.decoder.forward(enc_z)
            dec_x=dec_x.to(self.device)
            err1=criterion(x,dec_x)

            distance1=criterion_dtw(x, dec_x)
            pred1=self.criticX.forward(x)
            criticx_loss=self.loss_fn(pred1)

            enc_x,_=self.decoder.forward(enc_z)
            enc_x=enc_x.to(self.device)
            dec_enc_z,_=self.encoder.forward(enc_x)
            dec_enc_z=dec_enc_z.to(self.device)

            pred2=self.criticZ.forward(enc_z)
            criticz_loss=self.loss_fn(pred2)

            err2=criterion(enc_z,dec_enc_z)
            distance2=criterion_dtw(enc_z, dec_enc_z)
            err=err1+err2+criticz_loss+criticx_loss
            losses.append(err.item())
        return losses