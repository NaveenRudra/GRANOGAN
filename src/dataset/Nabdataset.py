import glob
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch
import json

class NabDataset(Dataset,):
    
    def __init__(self,data_settings):
        super().__init__()
        self.train=data_settings.train
        df_x,df_y = self.read_data(data_folder_path = data_settings.data_folder_path,
                                   label_file=data_settings.label_file,
                                   key=data_settings.key
                                   )   
        
        df_x = df_x[['value']]
        df_x=self.normalize(df_x)
        df_x.columns=['value']
        
        self.window_length = data_settings.window_length
        
        if data_settings.train:
            self.stride=1
        else:
            self.stride=self.window_length
            
        self.n_feature = len(df_x.columns)
        
        x=df_x
        y=df_y
        
        # adapt the datasets for the sequence data shape
        x,y = self.unroll(x,y)
        
        #The below line creates .. Creates a Tensor from a numpy.ndarray 
        self.x = torch.from_numpy(x).float()
        self.y=torch.from_numpy(np.array([1 if sum(y_i) > 0 else 0 for y_i in y])).float()
        
        self.data_len = x.shape[0]        
      
    def __len__(self):
        return self.data_len
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  
        
       # create sequences 
    def unroll(self, data, labels):
        un_data = []
        un_labels = []
        seq_len = int(self.window_length)
        stride = int(self.stride)
        
        idx = 0
        while(idx < len(data) - seq_len):
            
            if self.train==1 and labels.loc[idx,0]==1.0:
              idx += stride
              continue
            un_data.append(data.iloc[idx:idx+seq_len].values)
            un_labels.append(labels.iloc[idx:idx+seq_len].values)
            idx += stride
        return np.array(un_data), np.array(un_labels)
    
    def read_data(self, data_folder_path=None, label_file=None, key=None, BASE=''):
        with open(label_file) as FI:
            j_label = json.load(FI)
        ano_spans = j_label[key]
        self.ano_span_count = len(ano_spans)
        
        all_files = glob.glob(data_folder_path + "/*.csv")

        dfs=[]
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            dfs.append(df)
        df_x = pd.concat(dfs, axis=0, ignore_index=True)
        
        df_x, df_y = self.assign_ano(ano_spans, df_x)
            
        return df_x, df_y
    
    def assign_ano(self, ano_spans=None, df_x=None):
        df_x['timestamp'] = pd.to_datetime(df_x['timestamp'])
        y = np.zeros(len(df_x))
        for ano_span in ano_spans:
            ano_start = pd.to_datetime(ano_span[0])
            ano_end = pd.to_datetime(ano_span[1])
            for idx in df_x.index:
                if df_x.loc[idx, 'timestamp'] >= ano_start and df_x.loc[idx, 'timestamp'] <= ano_end:
                    y[idx] = 1.0
        return df_x, pd.DataFrame(y)
    
    def normalize(self, df_x=None):
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(df_x)
        df_x = pd.DataFrame(np_scaled)
        return df_x 