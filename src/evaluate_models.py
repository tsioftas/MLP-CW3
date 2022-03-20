import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

def load_data():
    print("Loading data...")

    input_dir = "data/"
    train_csv = 'shifts_canonical_dev_in.csv' # 'shifts_canonical_train.csv'
    val_indom_csv = 'shifts_canonical_dev_in.csv'
    val_outdom_csv = 'shifts_canonical_dev_out.csv'
    eval_indom_csv = 'shifts_canonical_eval_in.csv'
    eval_outdom_csv = 'shifts_canonical_eval_out.csv'

    print('Loading training data...')
    train_df = pd.read_csv(input_dir+train_csv)
    x_train = train_df[train_df.columns.drop(['climate'] + list(train_df.filter(regex='fact_')))].astype(np.float32)
    y_train = train_df['fact_temperature'].astype(np.float32).to_frame()

    print('Loading in-domain evaluation data...')
    eval_indom_df = pd.read_csv(input_dir+eval_indom_csv)
    x_eval_indom = eval_indom_df[eval_indom_df.columns.drop(['climate'] + list(eval_indom_df.filter(regex='fact_')))].astype(np.float32)
    y_eval_indom = eval_indom_df['fact_temperature'].to_frame()
    
    print('Loading out-domain evaluation data...')
    eval_outdom_df = pd.read_csv(input_dir+eval_outdom_csv)
    x_eval_outdom = eval_outdom_df[eval_outdom_df.columns.drop(['climate'] + list(eval_outdom_df.filter(regex='fact_')))].astype(np.float32)
    y_eval_outdom = eval_outdom_df['fact_temperature'].to_frame()

    return x_train, y_train, x_eval_indom, y_eval_indom, x_eval_outdom, y_eval_outdom

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

def preprocess_testData(df, mean=None, std=None):
    if mean is None:
        mean = df['fact_temperature'].mean()
    if std is None:
        std = df['fact_temperature'].std()
    df['fact_temperature'] = (df['fact_temperature'] - mean) / std
    return df, mean, std

def un_normalize(df, mean, std):
    df = df * std + mean
    return df

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

def pca_pre(tr, va, te, n_comp, feat_new):
    print("Applying PCA to features...")
    pca = PCA(n_components=n_comp, random_state=42)
    tr2 = pd.DataFrame(pca.fit_transform(tr),columns=feat_new)
    va2 = pd.DataFrame(pca.transform(va),columns=feat_new)
    te2 = pd.DataFrame(pca.transform(te),columns=feat_new)
    print("Finished applying PCA to features")
    return(tr2,va2,te2)

def actual_preprocess_data(data, n_comp):
    print("Started data pre-processing...")
    x_train, y_train, x_eval_indom, y_eval_indom, x_eval_outdom, y_eval_outdom = data
    
    x_train = preprocess(x_train)
    y_train, mean_train, std_train = preprocess_testData(y_train)
    x_eval_indom = preprocess(x_eval_indom)
    y_eval_indom, _, _ = preprocess_testData(y_eval_indom, mean_train, std_train)
    x_eval_outdom = preprocess(x_eval_outdom)
    y_eval_outdom, _, _ = preprocess_testData(y_eval_outdom, mean_train, std_train)
    x_train, ss     = norm_fit(x_train, True, 'quan')
    x_eval_indom  = norm_tra(x_eval_indom, ss)
    x_eval_outdom = norm_tra(x_eval_outdom, ss)

    pca_feat = [f'pca_-{i}' for i in range(n_comp)]
    x_train_pca, x_eval_indom_pca, x_eval_outdom_pca = pca_pre(x_train, x_eval_indom, x_eval_outdom, n_comp, pca_feat)
    x_train = pd.concat([x_train, x_train_pca],axis = 1)
    x_eval_indom = pd.concat([x_eval_indom, x_eval_indom_pca],axis = 1)
    x_eval_outdom  = pd.concat([x_eval_outdom, x_eval_outdom_pca],axis = 1)
    print("Finished data pre-processing")
    return x_train, y_train, x_eval_indom, y_eval_indom, x_eval_outdom, y_eval_outdom, mean_train, std_train

def load_model(path, num_features, num_targets, hidden_size):
    model = Model(num_features, num_targets, hidden_size)
    model.load_state_dict(torch.load(path))
    return model

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

def eval_fn(model, loss_fn, dataloader, device):
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

def main():
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
    n_comp = 40
    path_to_model = f'src/model_cnn_deep_lr=1-5_{n_comp}comp.pth'
    loss_fn = torch.nn.MSELoss().to('cuda')

    # Load data
    data = load_data()
    x_train, y_train, x_eval_indom, y_eval_indom, x_eval_outdom, y_eval_outdom, mean_train, std_train = actual_preprocess_data(data, n_comp)
    # train_dataset = TrainDataset(x_train, y_train)
    eval_indom_dataset = TrainDataset(x_eval_indom, y_eval_indom)
    eval_outdom_dataset = TrainDataset(x_eval_outdom, y_eval_outdom)
    #trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_indom_dataloader = torch.utils.data.DataLoader(eval_indom_dataset, batch_size=BATCH_SIZE, shuffle=False)
    eval_outdom_dataloader = torch.utils.data.DataLoader(eval_outdom_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define model based on data
    feature_cols= x_train.columns.values.tolist()
    target_cols = ['fact_temperature']
    num_features=len(feature_cols)  
    num_targets=len(target_cols)
    hidden_size=512

    # Load the model
    model = load_model(path_to_model, num_features, num_targets, hidden_size)
    model.to(DEVICE)

    eval_loss_indom, in_outputs = eval_fn(model, loss_fn, eval_indom_dataloader, DEVICE)
    # eval_loss_outdom, out_outputs = eval_fn(model, loss_fn, eval_outdom_dataloader, DEVICE)
    print(f"Eval loss indom: {eval_loss_indom}")
    # print(f"Eval loss outdom: {eval_loss_outdom}")
    
    in_outputs = un_normalize(in_outputs, mean_train, std_train)
    # out_outputs = un_normalize(out_outputs)

    marker = ','

    plt.title("In-domain labels and predictions")
    plt.ylabel('fact_temperature')
    plt.xlabel('datapoint-index')
    m_in_labels, _ = x_eval_indom.shape
    x_labels = [i for i in range(m_in_labels)]
    y_labels = y_eval_indom
    m_in_preds = len(in_outputs)
    x_preds = [i for i in range(m_in_preds)]
    y_preds = in_outputs
    plt.scatter(x_labels, y_labels, c='r', label='Labels', marker=marker)
    plt.scatter(x_preds, y_preds, c='b', label='Predictions', marker=marker)
    plt.legend()
    plt.savefig('in-domain-scatter.png')

    plt.clf()
    
    # plt.title("Out-of-domain labels and predictions")
    # plt.ylabel('fact_temperature')
    # plt.xlabel('datapoint-index')
    # m_out_labels, _ = x_eval_outdom.shape
    # x_labels = [i for i in range(m_out_labels)]
    # y_labels = y_eval_outdom
    # m_out_preds = len(out_outputs)
    # x_preds = [i for i in range(m_out_preds)]
    # y_preds = out_outputs
    # plt.scatter(x_labels, y_labels, c='r', label='Labels', marker=marker)
    # plt.scatter(x_preds, y_preds, c='b', label='Predictions', marker=marker)
    # plt.legend()
    # plt.savefig('out-doman-scatter.png')


if __name__ == "__main__":
    main()