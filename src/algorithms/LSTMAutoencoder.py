import torch
import torch.nn as nn 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import datetime
from src.model.modelLSTM import Encoder,Decoder,LSTMAutoencoder
from src.utils.losses import Wasserstein
import torch.nn.init as init
from src.utils.util import *
from collections import OrderedDict
from src.utils.tf_dtw import SoftDTW

class LSTMAutoencoderAlgo:
    
    def __init__(self,device=None,opt_trn=None,windows_length=60,n_features=1,embedding_dim=16):
        self.embedding_dim=embedding_dim
        self.device=device
        self.lr=opt_trn.lr
        self.windows_length=windows_length
        self.in_dim=n_features
        self.epochs=opt_trn.epochs
        self.criterion = torch.nn.MSELoss()
        self.autoencoder=LSTMAutoencoder(embedding_dim, n_features,device=device)
        self.autoencoder=nn.DataParallel(self.autoencoder)
        self.autoencoder=self.autoencoder.to(self.device)
        self.optimizerAutoEncoder = optim.Adam(self.autoencoder.parameters() , lr=self.lr)

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
    
    def intialize_lstmautoencoder(self,autoencoder):
        self.autoencoder=autoencoder
        
    def predict_loss(self,sequences):
        losses = []
        for x in sequences:
            x=x.float().to(self.device)
            output=self.autoencoder.forward(x)
            err=self.criterion(x,output.to(self.device))
            losses.append(err.item())
        return losses
    
    def train_autoencoder(self,sequences):
        history = dict(train=[], val=[])
        for epoch in range(0,self.epochs):
            train_losses = []
            for x in sequences:
                self.optimizerAutoEncoder.zero_grad()
                x=x.float().to(self.device)
                output=self.autoencoder.forward(x)
                err=self.criterion(x,output.to(self.device))
                err.backward()
                self.optimizerAutoEncoder.step()
                train_losses.append(err.item())
            train_loss = np.mean(train_losses)
            history['train'].append(train_loss)    
            print(f'Epoch {epoch}: train loss {train_loss}')
        return self.autoencoder