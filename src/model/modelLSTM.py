import torch
import torch.nn as nn 


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
       
        
    def forward(self,input):
        
        batch_size,seq_len=input.size(0),input.size(1)
        
        #h is the hidden state at time t and c is the cell state at time t
        h_0 = torch.zeros(1, batch_size, 2 * self.embedding_dim).to(self.device)
        c_0 = torch.zeros(1, batch_size, 2 * self.embedding_dim).to(self.device)
        
        recurrent_features,(h_1,c_1) = self.lstm1(input,(h_0,c_0))
        recurrent_features,_ = self.lstm2(recurrent_features)
        
        outputs=recurrent_features.view(batch_size,seq_len,self.embedding_dim)
        
        return outputs,recurrent_features
    

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
    
class LSTMAutoencoder(nn.Module):
    def __init__(self,  embedding_dim=16, n_features=1,device=None):
        super(LSTMAutoencoder, self).__init__()
        self.embedding_dim=embedding_dim
        self.n_features=n_features
        self.encoder = Encoder(self.n_features, 16,device)
        self.decoder=Decoder(16,self.n_features,device)
        
    def forward(self,input):
        enc_output,rec_features = self.encoder(input)
        dec_output,rec_features=self.decoder(enc_output)
        
        return dec_output