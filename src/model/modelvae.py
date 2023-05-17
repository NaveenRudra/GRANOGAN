import torch
import torch.nn as nn 
from torch.nn import functional as F

#https://github.com/vincrichard/LSTM-AutoEncoder-Unsupervised-Anomaly-Detection/blob/master/src/model/LSTM_auto_encoder.py 
#https://github.com/CUN-bjy/lstm-vae-torch/blob/main/src/models.py
#https://github.com/CUN-bjy/lstm-vae-torch/blob/main/src/models.py

#Performance time taken.

class Encoder(nn.Module):
    
    def __init__(self,n_features,embedding_dim=16,device=None):
        super(Encoder,self).__init__()
        self.in_dim=n_features
        self.device=device
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.lstm1 = nn.LSTM(
                    input_size=n_features,
                    hidden_size=self.hidden_dim,
                    num_layers=1,
                    batch_first=True
                    )
        self.lstm2 = nn.LSTM(
                    input_size=self.hidden_dim,
                    hidden_size=embedding_dim,
                    num_layers=1,
                    batch_first=True
                    )
        self.embed_2_mu=nn.Linear(self.hidden_dim,self.embedding_dim)
        self.embed_2_sigma=nn.Linear(self.hidden_dim,self.embedding_dim)
        
    def forward(self,input):
        
        batch_size,seq_len=input.size(0),input.size(1)
        
        #h is the hidden state at time t and c is the cell state at time t
        h_0 = torch.zeros(1, batch_size, 2 * self.embedding_dim).to(self.device)
        c_0 = torch.zeros(1, batch_size, 2 * self.embedding_dim).to(self.device)
        
        recurrent_features,(h_1,c_1) = self.lstm1(input,(h_0,c_0))
        mu=recurrent_features.view(batch_size,seq_len,self.hidden_dim)
        sigma=recurrent_features.view(batch_size,seq_len,self.hidden_dim)

        outputs_mu = self.embed_2_mu(mu)
        outputs_sigma=self.embed_2_sigma(sigma)
        noise = torch.randn(batch_size, seq_len, self.embedding_dim).to(self.device)

        z = noise * outputs_sigma + outputs_mu

        return z,mu,sigma,recurrent_features
    
    
class Decoder(nn.Module):
    def __init__(self,  embedding_dim=16, n_features=1,device=None):
        super(Decoder, self).__init__()
        self.device=device
        self.n_features=n_features
        self.hidden_dim, self.embedding_dim = 2 * embedding_dim, embedding_dim 
        self.lstm1 = nn.LSTM(
                    input_size=self.embedding_dim,
                    hidden_size=self.hidden_dim,
                    num_layers=1,
                    batch_first=True
                    )
        self.lstm2 = nn.LSTM(
        input_size=self.hidden_dim,
        hidden_size=self.n_features,
        num_layers=1,
        batch_first=True
        )
        
        
    def forward(self, input):
        batch_size,seq_len=input.size(0),input.size(1)
        h_0 = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        
        
        recurrent_features,(h_1,c_1)=self.lstm1(input,(h_0,c_0))
        recurrent_features,_ = self.lstm2(recurrent_features)
 
        outputs=recurrent_features.view(batch_size, seq_len, self.n_features)
        return outputs,recurrent_features
    
    
class VAE(nn.Module):
    def __init__(self,  embedding_dim=16, n_features=1,device=None):
        super(VAE, self).__init__()
        self.device=device
        self.embedding_dim=embedding_dim
        self.n_features=n_features
        self.encoder = Encoder(self.n_features, 16,self.device)
        self.decoder=Decoder(16,self.n_features,self.device)

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = 0.00025  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }


    def forward(self,input):
        z,mu,sigma,recurrent_features = self.encoder(input)
        dec_output,rec_features=self.decoder(z)
        losses = self.loss_function(dec_output, input, mu, sigma)
        m_loss, recon_loss, kld_loss = (
            losses["loss"],
            losses["Reconstruction_Loss"],
            losses["KLD"],
        )
        
        return m_loss, dec_output, (recon_loss, kld_loss)