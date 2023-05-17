
import torch
import torch.nn as nn 

#Remove anomalieis and train the data

class LSTMGenerator(nn.Module):
    
    def __init__(self,in_dim,out_dim,device=None):
        super().__init__()
        self.out_dim=out_dim
        self.in_dim=in_dim
        self.device=device
        
        self.lstm0=nn.LSTM(in_dim,hidden_size=32,num_layers=1,batch_first=True)
        self.lstm1=nn.LSTM(input_size=32,hidden_size=64,num_layers=1,batch_first=True)
        self.lstm2=nn.LSTM(input_size=64,hidden_size=128,num_layers=1,batch_first=True)
        
        self.linear=nn.Sequential(nn.Linear(in_features=128,out_features=out_dim),nn.Tanh())
        
    def forward(self,input):
        #Note LSTM works with multiple dimesional data
        batch_size,seq_len=input.size(0),input.size(1)
        #h is the hidden state at time t and c is the cell state at time t
        h_0 = torch.zeros(1, batch_size, 32).to(self.device)
        c_0 = torch.zeros(1, batch_size, 32).to(self.device)
        
        recurrent_features,(h_1,c_1) = self.lstm0(input,(h_0,c_0))
        recurrent_features,(h_2,c_2) = self.lstm1(recurrent_features)
        # According to Python doc, the special identifier _ is used in the interactive interpreter to store the result of the last evaluation.
        recurrent_features,_ = self.lstm2(recurrent_features)
        #contiguous().view is needed because any changes made to recurrent_features will not affect the ouput
        outputs=self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 128))
        outputs=outputs.view(batch_size,seq_len,self.out_dim)
        
        return outputs,recurrent_features
    
class LSTMDiscriminator(nn.Module):
    
    def tuple_of_tensors_to_tensor(tuple_of_tensors):
        return  torch.stack(list(tuple_of_tensors), dim=0)
    
    def __init__(self,in_dim,device=None):
        super().__init__()
        self.device=device
        
        self.in_dim=in_dim
        self.lstm=nn.LSTM(input_size=in_dim,hidden_size=100,num_layers=1,batch_first=True) #LSTM by deafult has tanh in it so nothing much to worry on this part
        self.linear=nn.Sequential(nn.Linear(100,1),nn.Sigmoid())
    
    def forward(self,input):
        batch_size,seq_len=input.size(0),input.size(1)
        h_0 = torch.zeros(1, batch_size, 100).to(self.device)
        c_0 = torch.zeros(1, batch_size, 100).to(self.device)
        
        recurrent_features,_=self.lstm(input,(h_0,c_0))
        outputs=self.linear(recurrent_features.contiguous().view(batch_size*seq_len, 100))
        outputs=outputs.view(batch_size, seq_len, 1)
        
        return outputs,recurrent_features
    