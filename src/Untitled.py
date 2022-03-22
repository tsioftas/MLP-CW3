#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
#from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import random
import matplotlib.pyplot as plt
import os
import copy
from copy import deepcopy as dp
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

# In[ ]:


def norm_fit(df_1,saveM = True, sc_name = 'zsco'):   
    from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,Normalizer,QuantileTransformer,PowerTransformer
    ss_1_dic = {'zsco':StandardScaler(),
                'mima':MinMaxScaler(),
                'maxb':MaxAbsScaler(), 
                'robu':RobustScaler(),
                'norm':Normalizer(), 
                'quan':QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal"),
                'powe':PowerTransformer()}
    ss_1 = ss_1_dic[sc_name]
    df_2 = pd.DataFrame(ss_1.fit_transform(df_1),index = df_1.index,columns = df_1.columns)
    if saveM == False:
        return(df_2)
    else:
        return(df_2,ss_1)

def norm_tra(df_1,ss_x):
    df_2 = pd.DataFrame(ss_x.transform(df_1),index = df_1.index,columns = df_1.columns)
    return(df_2)

def g_table(list1):
    table_dic = {}
    for i in list1:
        if i not in table_dic.keys():
            table_dic[i] = 1
        else:
            table_dic[i] += 1
    return(table_dic)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)


# In[ ]:


class SimpleCNN(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        cha_1 = 256 // 8
        cha_2 = 512 // 8
        cha_3 = 512 // 8

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        # self.batch_norm1 = nn.BatchNorm1d(num_features)
        # self.dropout1 = nn.Dropout(0.1)
        # self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        # self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        # self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        # self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        # self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        # self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        # self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        # self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        # self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        # self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        # self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        # self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        # self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        # x = self.batch_norm1(x)
        # x = self.dropout1(x)
        # x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                          self.cha_1_reshape)

        # x = self.batch_norm_c1(x)
        # x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        # x = self.ave_po_c1(x)

        # x = self.batch_norm_c2(x)
        # x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        # x = self.batch_norm_c2_1(x)
        # x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        # x = self.batch_norm_c2_2(x)
        # x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        # x = self.max_po_c2(x)

        x = self.flt(x)

        # x = self.batch_norm3(x)
        # x = self.dropout3(x)
        x = self.dense3(x)

        return x

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        cha_1 = 256 // 8
        cha_2 = 512 // 8
        cha_3 = 512 // 8

        cha_1_reshape = int(hidden_size/cha_1)
        cha_po_1 = int(hidden_size/cha_1/2)
        cha_po_2 = int(hidden_size/cha_1/2/2) * cha_3

        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3
        self.cha_1_reshape = cha_1_reshape
        self.cha_po_1 = cha_po_1
        self.cha_po_2 = cha_po_2

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1,cha_2, kernel_size = 5, stride = 1, padding=2,  bias=False),dim=None)

        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size = cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_2, kernel_size = 3, stride = 1, padding=1, bias=True),dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2,cha_3, kernel_size = 5, stride = 1, padding=2, bias=True),dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))

    def forward(self, x):

        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0],self.cha_1,
                          self.cha_1_reshape)

        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))

        x = self.ave_po_c1(x)

        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x =  x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


# In[ ]:


#tar_freq = np.array([np.min(list(g_table(train[target_cols].iloc[:,i]).values())) for i in range(len(target_cols))])
#tar_weight0 = np.array([np.log(i+100) for i in tar_freq])
#tar_weight0_min = dp(np.min(tar_weight0))
#tar_weight = tar_weight0_min/tar_weight0
#pos_weight = torch.tensor(tar_weight).to(DEVICE)
from torch.nn.modules.loss import _WeightedLoss
class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets, n_labels, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight,
                                                      pos_weight = None)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


# In[ ]:


class TrainDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features.iloc[idx, :].values, dtype=torch.float).to('cuda'),
            'y' : torch.tensor(self.targets.iloc[idx, :].values, dtype=torch.float).to('cuda')     
        }
        return dct


# In[ ]:


class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features.iloc[idx, :].values, dtype=torch.float).to('cuda')
        }
        return dct


# In[ ]:


def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)
        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


# In[ ]:


def preprocess(df):
    # * imputing: the missing values are replaced in all input columns following a simple constant strategy (fill value is −1);
    # * quantization: each input column is discretized into quantile bins, and the number of these bins is detected
    # automatically; after that the bin identifier can be considered quantized numerical value of the original feature;
    # * standardization: each quantized column is standardized by removing the mean and scaling to unit variance;
    # * decorrelation: all possible linear correlations are removed from the feature vectors discretized by above-mentioned way;
    # the decorrelation is implemented using PCA
    df.fillna(-1, inplace = True)
    
    q2 = df.apply(np.quantile,axis = 1,q = 0.25).copy()
    q7 = df.apply(np.quantile,axis = 1,q = 0.75).copy()
    qmean = (q2+q7)/2
    df = (df.T - qmean.values).T
    
    return df


# In[ ]:


def preprocess_testData(df):
    df['fact_temperature'] = (df['fact_temperature'] - df['fact_temperature'].mean()) / df['fact_temperature'].std()
    return df


# In[ ]:


def pca_pre(tr, va, te, n_comp, feat_new):
    pca = PCA(n_components=n_comp, random_state=42)
    tr2 = pd.DataFrame(pca.fit_transform(tr),columns=feat_new)
    va2 = pd.DataFrame(pca.transform(va),columns=feat_new)
    te2 = pd.DataFrame(pca.transform(te),columns=feat_new)
    return(tr2,va2,te2)

def try_different_pca_comb(tr):
    best_component = 0
    best_variance = 0
    for i in range(20,80):
        pca = PCA(n_components=i, random_state=42)
        principalComponents = pca.fit_transform(tr)
        retainedVariance = round(sum(list(pca.explained_variance_ratio_))*100, 2)
        print(i,retainedVariance)
        if retainedVariance < best_variance:
            best_variance = retainedVariance
            best_component = i
    print(best_component,best_variance)





# In[ ]:


SEED = [0, 1, 2, 3 ,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
input_dir = '../data/'

train_df = pd.read_csv(input_dir+'shifts_canonical_train.csv')
x_train = train_df[train_df.columns.drop(['climate'] + list(train_df.filter(regex='fact_')))].astype(np.float32)
y_train = train_df['fact_temperature'].astype(np.float32).to_frame()

print('1')
val_df_inDom = pd.read_csv(input_dir+'shifts_canonical_dev_in.csv')
x_valid_inDom = val_df_inDom[val_df_inDom.columns.drop(['climate'] + list(val_df_inDom.filter(regex='fact_')))].astype(np.float32)
y_valid_inDom = val_df_inDom['fact_temperature'].to_frame()
print('2')
val_df_outDom = pd.read_csv(input_dir+'shifts_canonical_dev_out.csv')
x_valid_outDom = val_df_outDom[val_df_outDom.columns.drop(['climate'] + list(val_df_outDom.filter(regex='fact_')))].astype(np.float32)
y_valid_outDom = val_df_outDom['fact_temperature'].to_frame()
print('3')

#test_df_inDom = pd.read_csv(input_dir+'shifts_canonical_eval_in.csv')
#x_test_inDom = test_df_inDom[test_df_inDom.columns.drop(['climate'] + list(test_df_inDom.filter(regex='fact_')))].astype(np.float32)
#y_test_inDom = test_df_inDom['fact_temperature'].to_frame()

#test_df_outDom = pd.read_csv(input_dir+'shifts_canonical_eval_out.csv')
#x_test_outDom = test_df_outDom[test_df_outDom.columns.drop(['climate'] + list(test_df_outDom.filter(regex='fact_')))].astype(np.float32)
#y_test_outDom = test_df_outDom['fact_temperature'].to_frame()


# In[ ]:


# HyperParameters
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False
seed = 42
n_comp = 22

feature_cols= x_train.columns.values.tolist()
target_cols = ['fact_temperature']
num_features=len(feature_cols) + n_comp
num_targets=len(target_cols)
#num_targets_0=len(target_nonsc_cols2)
hidden_size=512


# In[ ]:


x_train = preprocess(x_train)
y_train = preprocess_testData(y_train)
print('4')
x_valid_inDom = preprocess(x_valid_inDom)
y_valid_inDom = preprocess_testData(y_valid_inDom)
print('5')
x_valid_outDom = preprocess(x_valid_outDom)
y_valid_outDom = preprocess_testData(y_valid_outDom)
print('6')
x_train,ss     = norm_fit(x_train,True,'quan')
x_valid_inDom  = norm_tra(x_valid_inDom,ss)
x_valid_outDom = norm_tra(x_valid_outDom,ss)
print('7')
try_different_pca_comb(x_train)

pca_feat = [f'pca_-{i}' for i in range(n_comp)]

x_tr_pca,x_vainDom_pca,x_vaoutDom_pca = pca_pre(x_train, x_valid_inDom, x_valid_outDom, n_comp, pca_feat)

x_train = pd.concat([x_train,x_tr_pca],axis = 1)
x_valid_inDom = pd.concat([x_valid_inDom, x_vainDom_pca],axis = 1)
x_valid_outDom  = pd.concat([x_valid_outDom, x_vaoutDom_pca],axis = 1)
print('8')
# x_test_inDom = x_test_inDom
# y_test_inDom

# x_test_outDom = x_test_outDom
# y_test_outDom


# In[ ]:


train_dataset = TrainDataset(x_train, y_train)
valid_dataset_inDom = TrainDataset(x_valid_inDom, y_valid_inDom)
valid_dataset_outDom = TrainDataset(x_valid_outDom, y_valid_outDom)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validloader_inDom = torch.utils.data.DataLoader(valid_dataset_inDom, batch_size=BATCH_SIZE, shuffle=False)
validloader_outDom = torch.utils.data.DataLoader(valid_dataset_outDom, batch_size=BATCH_SIZE, shuffle=False)

print('9')
# In[ ]:


model = SimpleCNN(
            num_features=num_features,
            num_targets=num_targets,
            hidden_size=hidden_size,
        )
model.to('cuda')

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, 
                                                  max_lr=1e-5, epochs=EPOCHS, steps_per_epoch=len(trainloader))

loss_tr = nn.MSELoss().to('cuda') #SmoothBCEwLogits(smoothing = 0.001)
loss_va_inDom = nn.MSELoss().to('cuda') #nn.BCEWithLogitsLoss()
loss_va_outDom = nn.MSELoss().to('cuda') #nn.BCEWithLogitsLoss()

early_stopping_steps = EARLY_STOPPING_STEPS
early_step = 0

oof = np.zeros(len(target_cols))
best_loss = np.inf

mod_name = f"FOLD_mod11_{seed}.pth"
print('10')
for epoch in range(EPOCHS):

    train_loss = train_fn(model, optimizer,scheduler, loss_tr, trainloader, DEVICE)
    valid_loss_inDom, valid_preds_inDom = valid_fn(model, loss_va_inDom, validloader_inDom, DEVICE)
    valid_loss_outDom, valid_preds_outDom = valid_fn(model, loss_va_outDom, validloader_outDom, DEVICE)
    print(f"SEED: {seed}, EPOCH: {epoch}, train_loss: {train_loss}, valid_loss_inDom: {valid_loss_inDom}, valid_loss_outDom: {valid_loss_outDom}")

    if valid_loss_outDom < best_loss:

        best_loss = valid_loss_outDom
        oof = valid_preds_outDom
        torch.save(model.state_dict(), mod_name)

    elif(EARLY_STOP == True):

        early_step += 1
        if (early_step >= early_stopping_steps):
            break


# In[ ]:





# In[ ]:




