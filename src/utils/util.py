import torch.autograd as autograd
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import List, Union
from torch.utils.data import DataLoader

def improve_detection(test_score_df):
    start_end = []
    state = 0
    for idx in test_score_df.index:
        if state==0 and test_score_df.loc[idx, 'y']==1:
            state=1
            start = idx
        if state==1 and test_score_df.loc[idx, 'y']==0:
            state = 0
            end = idx
            start_end.append((start, end))

    for s_e in start_end:
        if sum(test_score_df[s_e[0]:s_e[1]+1]['anomaly'])>0:
            for i in range(s_e[0], s_e[1]+1):
                test_score_df.loc[i, 'anomaly'] = 1

    actual = np.array(test_score_df['y'])
    predicted = np.array([int(a) for a in test_score_df['anomaly']])
    return actual,predicted
    
    
def print_scores(predicted,actual):   
    tp = np.count_nonzero(predicted * actual)
    tn = np.count_nonzero((predicted - 1) * (actual - 1))
    fp = np.count_nonzero(predicted * (actual - 1))
    fn = np.count_nonzero((predicted - 1) * actual)

    print('True Positive\t', tp)
    print('True Negative\t', tn)
    print('False Positive\t', fp)
    print('False Negative\t', fn)

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall)
    cohen_k_score = cohen_kappa_score(predicted, actual)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predicted)
    auc_val = auc(false_positive_rate, true_positive_rate)
    roc_auc_val = roc_auc_score(actual, predicted)

    print('Accuracy\t', accuracy)
    print('Precision\t', precision)
    print('Recall\t', recall)
    print('f-measure\t', fmeasure)
    print('cohen_kappa_score\t', cohen_k_score)
    print('auc\t', auc_val)
    print('roc_auc\t', roc_auc_val)
    

def normalize_gradient(net_D, x, **kwargs):
    
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    x.requires_grad_(True)
    f = net_D(x, **kwargs)
    grad = torch.autograd.grad(
        f, [x], torch.ones_like(f), create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=2), p=2, dim=2)
    grad_norm = grad_norm.view(-1, *[1 for _ in range(len(f.shape) - 1)])
    
    f_hat = (f / (grad_norm + torch.abs(f)))
    return f_hat

def get_sub_seqs(x_arr, y_arr, seq_len, stride=1, start_discont=np.array([])):
    """
    :param start_discont: the start points of each sub-part in case the x_arr is just multiple parts joined together
    :param x_arr: dim 0 is time, dim 1 is channels
    :param seq_len: size of window used to create subsequences from the data
    :param stride: number of time points the window will move between two subsequences
    :return:
    """
    excluded_starts = []
    [excluded_starts.extend(range((start - seq_len + 1), start)) for start in start_discont if start > seq_len]
    seq_starts = np.delete(np.arange(0, x_arr.shape[0] - seq_len + 1, stride), excluded_starts)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])
    y_seqs=  np.array([y_arr[i:i + seq_len] for i in seq_starts])
    y=torch.from_numpy(np.array([1 if sum(y_i) > 0 else 0 for y_i in y_seqs])).float()
    return x_seqs,y


def get_train_data_loaders(x_seqs: np.ndarray, batch_size: int, splits: List, seed: int, shuffle: bool = False,
    usetorch = True):
    """
    Splits the train data between train, val, etc. Creates and returns pytorch data loaders
    :param shuffle: boolean that determines whether samples are shuffled before splitting the data
    :param seed: seed used for the random shuffling (if shuffling there is)
    :param x_seqs: input data where each row is a sample (a sequence) and each column is a channel
    :param batch_size: number of samples per batch
    :param splits: list of split fractions, should sum up to 1.
    :param usetorch: if True returns dataloaders, otherwise return datasets
    :return: a tuple of data loaders as long as splits. If len_splits = 1, only 1 data loader is returned
    """
    if np.sum(splits) != 1:
        scale_factor = np.sum(splits)
        splits = [fraction/scale_factor for fraction in splits]
    if shuffle:
        np.random.seed(seed)
        x_seqs = x_seqs[np.random.permutation(len(x_seqs))]
        np.random.seed()
    split_points = [0]
    for i in range(len(splits)-1):
        split_points.append(split_points[-1] + int(splits[i]*len(x_seqs)))
    split_points.append(len(x_seqs))
    if usetorch:
        loaders = tuple([DataLoader(dataset=x_seqs[split_points[i]:split_points[i+1]], batch_size=batch_size,
            drop_last=False, pin_memory=True, shuffle=False) for i in range(len(splits))])
        return loaders
    else:
        # datasets = tuple([x_seqs[split_points[i]: 
        #     (split_points[i] + (split_points[i+1]-split_points[i])//batch_size*batch_size)] 
        #     for i in range(len(splits))])
        datasets = tuple([x_seqs[split_points[i]:split_points[i+1]]
            for i in range(len(splits))])
        return datasets