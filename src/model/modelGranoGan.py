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
    

    
#This is also the Generator

class Decoder(nn.Module):
    """An LSTM based generator. It expects a sequence of noise vectors as input.
    Args:
        in_dim: Input noise dimensionality
        out_dim: Output dimensionality
        n_layers: number of lstm layers
        hidden_dim: dimensionality of the hidden layer of lstms
    Input: noise of shape (batch_size, seq_len, in_dim)
    Output: sequence of shape (batch_size, seq_len, out_dim)
    """

    def __init__(self, in_dim, out_dim, device=None):
        super().__init__()
        self.out_dim = out_dim
        self.device = device

        self.lstm0 = nn.LSTM(in_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        
        self.linear = nn.Sequential(nn.Linear(in_features=128, out_features=out_dim), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(1, batch_size, 32).to(self.device)
        c_0 = torch.zeros(1, batch_size, 32).to(self.device)

        recurrent_features, (h_1, c_1) = self.lstm0(input, (h_0, c_0))
        recurrent_features, (h_2, c_2) = self.lstm1(recurrent_features)
        recurrent_features, _ = self.lstm2(recurrent_features)
        
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 128))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs, recurrent_features
    
    
    
class Critic(nn.Module):

    def __init__(self,in_dim,device=None):
        super().__init__()
        self.device=device

        self.lstm=nn.LSTM(input_size=in_dim,hidden_size=100,num_layers=1,batch_first=True)
        self.linear = nn.Sequential(nn.Linear(100,1))

    def forward(self,input):
        batch_size,seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(1, batch_size, 100).to(self.device)
        c_0 = torch.zeros(1, batch_size, 100).to(self.device)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 100))
        return outputs