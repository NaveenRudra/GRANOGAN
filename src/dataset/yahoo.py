import glob
from torch.utils.data import Dataset
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch
        
class YahooDataset(Dataset):
    def __init__(self, data_settings):
        self.train = data_settings.train
        df_x, df_y = self.read_data(data_folder_path = data_settings.data_folder_path)
        #select and standardize data
        df_x = df_x[['value']]
        df_x = self.normalize(df_x)
        df_x.columns = ['value']
        
        # important parameters
        #self.window_length = int(len(df_x)*0.1/self.ano_span_count)
        self.window_length = 60
        if data_settings.train:
            self.stride = 1
        else:
            self.stride = self.window_length
        
        self.n_feature = len(df_x.columns)
        
        # x, y data
        x = df_x
        y = df_y

        # adapt the datasets for the sequence data shape
        x, y = self.unroll(x, y)
        
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(np.array([1 if sum(y_i) > 0 else 0 for y_i in y])).float()
        
        
        self.data_len = x.shape[0]
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    # create sequences 
    def unroll(self, data, labels):
        un_data = []
        un_labels = []
        seq_len = int(self.window_length)
        stride = int(self.stride)
        
        idx = 0
        while(idx < len(data) - seq_len):
            '''
            if self.train==1 and labels.loc[idx,'is_anomaly']==1.0:
                idx += stride
                continue
            '''
            un_data.append(data.iloc[idx:idx+seq_len].values)
            un_labels.append(labels.iloc[idx:idx+seq_len].values)
            idx += stride
        return np.array(un_data), np.array(un_labels)
    
    def read_data(self, data_folder_path=None):
        all_files = glob.glob(data_folder_path + "/*.csv")

        dfs=[]
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            dfs.append(df)
        df_x = pd.concat(dfs, axis=0, ignore_index=True)
       
        #y=np.zeros(len(df_x))
        y=df_x['is_anomaly']
        df_x=df_x.drop(['is_anomaly'], axis=1)
        return df_x, pd.DataFrame(y)
    
    def normalize(self, df_x=None):
        min_max_scaler = preprocessing.StandardScaler()
        np_scaled = min_max_scaler.fit_transform(df_x)
        df_x = pd.DataFrame(np_scaled)
        return df_x