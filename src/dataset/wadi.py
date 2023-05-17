import math
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

class WadiDataset(Dataset,):
    def __init__(self,seed: int,data_settings=None,remove_unique=False, entity=None, verbose=False, one_hot=False):
        super().__init__()
        self._data = None
        self.one_hot=one_hot
        self.verbose=verbose
        self.remove_unique=remove_unique
        self.train=data_settings.train
        self.dataset_training_name =data_settings.dataset_training_name
        self.dataset_test_name =data_settings.dataset_test_name
        self.dataset_anomaly_name =data_settings.dataset_anomaly_name
        self.window_length=data_settings.window_length
        if data_settings.train:
            self.stride=1
        else:
            self.stride=self.window_length
        
        
        
        '''
        x,y=self.load()
        self.n_feature = len(x.columns)
        x,y=self.unroll(x[:1000],y[:1000])
        self.x = torch.from_numpy(x).float()
        self.y=torch.from_numpy(np.array([1 if sum(y_i) > 0 else 0 for y_i in y])).float()
        
        self.data_len = x.shape[0]  
       '''
    def data(self) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """Return data, load if necessary"""
        if self._data is None:
            self.load()
        return self._data    
         
    
    def __len__(self):
        return self.data_len
    
    def standardize(self,X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4):

        mini = X_train.min()
        maxi = X_train.max()
        for col in X_train.columns:
            if maxi[col] != mini[col]:
                X_train[col] = (X_train[col] - mini[col]) / (maxi[col] - mini[col])
                X_test[col] = (X_test[col] - mini[col]) / (maxi[col] - mini[col])
                X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
            else:
                assert X_train[col].nunique() == 1
                if remove:
                    if verbose:
                        print("Column {} has the same min and max value in train. Will remove this column".format(col))
                    X_train = X_train.drop(col, axis=1)
                    X_test = X_test.drop(col, axis=1)
                else:
                    if verbose:
                        print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                    if mini[col] != 0:
                        X_train[col] = X_train[col] / mini[col]  # Redundant operation, just for consistency
                        X_test[col] = X_test[col] / mini[col]
                    if verbose:
                        print("After transformation, train unique vals: {}, test unique vals: {}".format(
                        X_train[col].unique(),
                        X_test[col].unique()))
        return X_train, X_test

    def format_data(self, train_df, test_df, OUTLIER_CLASS=1, verbose=False):
        train_only_cols = set(train_df.columns).difference(set(test_df.columns))
        if verbose:
            print("Columns {} present only in the training set, removing them")
        train_df = train_df.drop(train_only_cols, axis=1)

        test_only_cols = set(test_df.columns).difference(set(train_df.columns))
        if verbose:
            print("Columns {} present only in the test set, removing them")
        test_df = test_df.drop(test_only_cols, axis=1)

        train_anomalies = train_df[train_df["y"] == OUTLIER_CLASS]
        test_anomalies: pd.DataFrame = test_df[test_df["y"] == OUTLIER_CLASS]
        print("Total Number of anomalies in train set = {}".format(len(train_anomalies)))
        print("Total Number of anomalies in test set = {}".format(len(test_anomalies)))
        print("% of anomalies in the test set = {}".format(len(test_anomalies) / len(test_df) * 100))
        print("number of anomalous events = {}".format(len(self.get_events(y_test=test_df["y"].values,outlier=1,normal=0))))
        # Remove the labels from the data
        X_train = train_df.drop(["y"], axis=1)
        y_train = train_df["y"]
        X_test = test_df.drop(["y"], axis=1)
        y_test = test_df["y"]
        self.y_test = y_test
        return X_train, y_train, X_test, y_test
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  
    
    def get_events(self,y_test, outlier=1, normal=0, breaks=[]):
        events = dict()
        label_prev = normal
        event = 0  # corresponds to no event
        event_start = 0
        for tim, label in enumerate(y_test):
            if label == outlier:
                if label_prev == normal:
                    event += 1
                    event_start = tim
                elif tim in breaks:
                    # A break point was hit, end current event and start new one
                    event_end = tim - 1
                    events[event] = (event_start, event_end)
                    event += 1
                    event_start = tim

            else:
                # event_by_time_true[tim] = 0
                if label_prev == outlier:
                    event_end = tim - 1
                    events[event] = (event_start, event_end)
            label_prev = label

        if label_prev == outlier:
            event_end = tim - 1
            events[event] = (event_start, event_end)
        return events
        
    def load(self):
        OUTLIER_CLASS = 1
        df_x_train: pd.DataFrame = pd.read_csv(self.dataset_training_name, header=3)
        df_x_test: pd.DataFrame = pd.read_csv(self.dataset_test_name, header=0)
        # Removing 4 columns who only contain nans (data missing from the csv file)
        nan_columns = [r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_001_AL',
                       r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_002_AL',
                       r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_001_STATUS',
                       r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_002_STATUS']        
        df_x_train=df_x_train.drop(nan_columns, axis=1)
        df_x_test = df_x_test.drop(nan_columns, axis=1)
        
        df_x_train = df_x_train.rename(columns={col: col.split('\\')[-1] for col in df_x_train.columns})
        df_x_test = df_x_test.rename(columns={col: col.split('\\')[-1] for col in df_x_test.columns})
        
        
        print(df_x_train.shape)
        ano_df = pd.read_csv(self.dataset_anomaly_name, header=0)
        df_x_train["y"] = np.zeros(df_x_train.shape[0])
        df_x_test["y"] = np.zeros(df_x_test.shape[0])
        
        pd.set_option('mode.chained_assignment', None) # This is to prevent error SettingWithCopyWarning:A value is trying to be set on a copy of a slice from a DataFrame
        for i in range(ano_df.shape[0]):
            ano = ano_df.iloc[i, :][["Start_time", "End_time", "Date"]]
            start_row = np.where((df_x_test["Time"].values == ano["Start_time"]) &
                                 (df_x_test["Date"].values == ano["Date"]))[0][0]
            end_row = np.where((df_x_test["Time"].values == ano["End_time"]) &
                               (df_x_test["Date"].values == ano["Date"]))[0][0]
            df_x_test["y"].iloc[start_row:(end_row + 1)] = np.ones(1 + end_row - start_row)
            
        df_x_train = df_x_train.drop(["Time", "Date", "Row"], axis=1)
        df_x_test = df_x_test.drop(["Time", "Date", "Row"], axis=1)
     
        if self.one_hot:
            # actuator colums (categoricals) with < 2 categories (all of these have 3 categories)
            one_hot_cols = ['1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS', '1_MV_004_STATUS', '2_MV_003_STATUS',
                            '2_MV_006_STATUS', '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS',
                            '2_MV_501_STATUS', '2_MV_601_STATUS']

            # combining before encoding because some categories only seen in test
            one_hot_encoded = Dataset.one_hot_encoding(pd.concat([df_x_train, df_x_test], axis=0, join="inner"),
                                                       col_names=one_hot_cols)
            df_x_train = one_hot_encoded.iloc[:len(df_x_train)]
            df_x_test = one_hot_encoded.iloc[len(df_x_train):]
        
        
        X_train, y_train, X_test, y_test = self.format_data(df_x_train, df_x_test, OUTLIER_CLASS, verbose=self.verbose)
        X_train, X_test = self.standardize(X_train, X_test)
        
        
        #X_train,y_train=self.unroll(X_train,y_train)
        
        #self.train=0
        #self.stride=self.window_length
        
        #X_test,y_test=self.unroll(X_test,y_test)
        self._data = tuple([X_train, y_train, X_test, y_test])
        #return X_train,y_train,X_test,y_test
        
    def unroll(self, data, labels):
        un_data = []
        un_labels = []
        seq_len = int(self.window_length)
        stride = int(self.stride)
        
        idx = 0
        while(idx < len(data) - seq_len):
            
            if self.train==1 and labels.loc[idx]==1.0:
              idx += stride
              continue
            un_data.append(data.iloc[idx:idx+seq_len].values)
            un_labels.append(labels.iloc[idx:idx+seq_len].values)
            idx += stride
        return np.array(un_data), np.array(un_labels)
    
    def get_root_causes(self):
        return self.causes