import math
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


class SwatDataset(Dataset,):

    def __init__(self, seed: int, shorten_long=True, remove_unique=False, entity=None, verbose=False, one_hot=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        self._data = None
        if shorten_long:
            name = "swat"
        else:
            name = "swat-long"
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                           "data",  "swat")
        self.raw_path_train = os.path.join(root, "SWaT_Dataset_Normal_v1.csv")
        self.raw_path_test = os.path.join(root, "SWaT_Dataset_Attack_v0.csv")

        if not os.path.isfile(self.raw_path_train):
            df = pd.read_excel(os.path.join(root, "SWaT_Dataset_Normal_v1.xlsx"))
            #removing first row as it is mentioned as unnamed
            df.to_csv(self.raw_path_train, index=False)
            
        if not os.path.isfile(self.raw_path_test):
            df = pd.read_excel(os.path.join(root, "SWaT_Dataset_Attack_v0.xlsx"))
            #removing first row as it is mentioned as unnamed
            df.to_csv(self.raw_path_test, index=False)

        self.seed = seed
        self.shorten_long = shorten_long
        self.remove_unique = remove_unique
        self.verbose = verbose
        self.one_hot = one_hot

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
        print("number of anomalous events = {}".format(len(self.get_events(y_test=test_df["y"].values))))
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
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        test_df: pd.DataFrame = pd.read_csv(self.raw_path_test)
        train_df: pd.DataFrame = pd.read_csv(self.raw_path_train)

        train_df = train_df.rename(columns={col: col.strip() for col in train_df.columns})
        test_df = test_df.rename(columns={col: col.strip() for col in test_df.columns})
        
        train_df.columns = train_df.iloc[0]
        test_df.columns = test_df.iloc[0]
        
        train_df.columns=train_df.columns.str.strip()
        test_df.columns=test_df.columns.str.strip()
        
        train_df=train_df.iloc[1:, :]
        test_df=test_df.iloc[1:, :]
        
        
        train_df["y"] = train_df["Normal/Attack"].replace(to_replace=["Normal", "Attack", "A ttack"], value=[0, 1, 1])
        train_df = train_df.drop(columns=["Normal/Attack", "Timestamp"], axis=1)
        test_df["y"] = test_df["Normal/Attack"].replace(to_replace=["Normal", "Attack", "A ttack"], value=[0, 1, 1])
        test_df = test_df.drop(columns=["Normal/Attack", "Timestamp"], axis=1)
        
        train_df=train_df.astype(float)
        test_df=test_df.astype(float)
        #print(train_df.head())
        
        # one-hot-encoding stuff
        if self.one_hot:
            keywords = {col_name: "".join([s for s in col_name if not s.isdigit()]) for col_name in train_df.columns}
            cat_cols = [col for col in keywords.keys() if keywords[col] in ["P", "MV", "UV"]]
            one_hot_cols = [col for col in cat_cols if train_df[col].nunique() >= 3 or test_df[col].nunique() >= 3]
            print(one_hot_cols)
            one_hot_encoded = Dataset.one_hot_encoding(pd.concat([train_df, test_df], axis=0, join="inner"),
                                                       col_names=one_hot_cols)
            train_df = one_hot_encoded.iloc[:len(train_df)]
            test_df = one_hot_encoded.iloc[len(train_df):]

        # shorten the extra long anomaly to 550 points
        if self.shorten_long:
            long_anom_start = 227828
            long_anom_end = 263727
            test_df = test_df.drop(test_df.loc[(long_anom_start + 551):(long_anom_end + 1)].index,
                                   axis=0).reset_index(drop=True)
        causes_channels_names = [["MV101"], ["P102"], ["LIT101"], [], ["AIT202"], ["LIT301"], ["DPIT301"],
                                 ["FIT401"], [], ["MV304"], ["MV303"], ["LIT301"], ["MV303"], ["AIT504"],
                                 ["AIT504"], ["MV101", "LIT101"], ["UV401", "AIT502", "P501"], ["P602", "DPIT301",
                                                                                                "MV302"],
                                 ["P203", "P205"], ["LIT401", "P401"], ["P101", "LIT301"], ["P302", "LIT401"],
                                 ["P201", "P203", "P205"], ["LIT101", "P101", "MV201"], ["LIT401"], ["LIT301"],
                                 ["LIT101"], ["P101"], ["P101", "P102"], ["LIT101"], ["P501", "FIT502"],
                                 ["AIT402", "AIT502"], ["FIT401", "AIT502"], ["FIT401"], ["LIT301"]]
        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, OUTLIER_CLASS, verbose=self.verbose)
        
        
        X_train, X_test = self.standardize(X_train, X_test, remove=self.remove_unique, verbose=self.verbose)

        matching_col_names = np.array([col.split("_1hot")[0] for col in train_df.columns])
        self.causes = []
        for event in causes_channels_names:
            event_causes = []
            for chan_name in event:
                event_causes.extend(np.argwhere(chan_name == matching_col_names).ravel())
            self.causes.append(event_causes)

        self._data = tuple([X_train, y_train, X_test, y_test])

    def get_root_causes(self):
        return self.causes

