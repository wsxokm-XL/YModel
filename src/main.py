import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

batch = 10
lr = 1e-3
num_epochs = 20

df = pd.read_csv("../nesg/nesg.csv")

fl = open("../nesg/nesg.fasta")
dic1 = {}
chro = fl.readline().strip().split('>')[1]
seq1 = ''
for i in fl:
    if '>' in i:
        dic1[chro] = seq1
        chro = i.strip().split('>')[1]
        seq1 = ''
    else:
        seq1 += i.strip()
dic1[chro] = seq1
fl.close()

for j in range(len(df)):
    if df.iloc[j, 2] > 0:
        df.iloc[j, 2] = 1

# for j in range(0, 5):
#     print(df.iloc[j, 0])    # id
#     print(dic1[df.iloc[j, 0]])  # seq
#     print(df.iloc[j, 1])    # exp
#     print(df.iloc[j, 2])    # sol
#     print()

# print(type(df['sol']))

# seq_length = []
# for j in range(len(df)):
#     seq_length.append(len(dic1[df.iloc[j, 0]]))

# print(max(seq_length))  # 979
# print(sum(seq_length))  # 2365465
# print(len(seq_length))  # 9703
# print(sum(seq_length)/len(seq_length))  # 243.787
# seq_length.sort(reverse=True)
# print(seq_length[2000])    # 355 => 360

seq = list(dic1.values())
df['seq'] = seq
data = df[['seq', 'sol']].values    # ndarray


def get_set(ddata):
    """return three list: test_set, train_set, valid_set"""
    f_set = []
    test_set = []   # 测试集：1941
    train_set = []  # 训练集：5821
    valid_set = []  # 验证集：1941
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for f_index, test_index in split.split(ddata[:, :-1], ddata[:, -1]):
        f_set = ddata[f_index, :]
        test_set = ddata[test_index, :]

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    for train_index, valid_index in split.split(f_set[:, :-1], f_set[:, -1]):
        train_set = f_set[train_index, :]
        valid_set = f_set[valid_index, :]

    return test_set, train_set, valid_set


def get_sol(d):
    """sol: return a tensor variable"""
    sol = d[:, 1]
    sol = sol.reshape(-1, 1)
    sol_onehotencoder = OneHotEncoder(sparse=False)
    sol_onehotencoded = sol_onehotencoder.fit_transform(sol)
    sol_onehotencoded = torch.from_numpy(sol_onehotencoded)
    return (sol_onehotencoded).float()


def get_seq(d):
    """given an array d, return a tensor with (d.shape[0],360,20)"""
    dic = {'X': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'A': (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'C': (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'D': (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'E': (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'F': (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'G': (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'H': (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'I': (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'K': (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'L': (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'M': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
           'N': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
           'P': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
           'Q': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
           'R': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
           'S': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
           'T': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
           'V': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
           'W': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
           'Y': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)}

    ds = np.zeros((1, 360, 20))
    for i in d[:, 0]:
        dds = np.zeros((1, 20))
        for j in i:
            dds = np.insert(dds, len(dds), dic[j], axis=0)
            dds = dds[1:dds.shape[0]]

            if dds.shape[0] <= 360:
                dds = np.pad(dds, ((0, 360 - dds.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
            else:
                dds = dds[0:360]

        ds = np.insert(ds, len(ds), dds, axis=0)

    ds = ds[1:ds.shape[0]]

    return (torch.tensor(ds)).float()


def get_data(d):
    x = get_seq(d)
    y = get_sol(d)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    return dl


test, train, valid = get_set(data)
# (pd.DataFrame(test, columns=['seq', 'sol'])).to_csv("../data/test_set.csv", index=0)
# (pd.DataFrame(train, columns=['seq', 'sol'])).to_csv("../data/train_set.csv", index=0)
# (pd.DataFrame(valid, columns=['seq', 'sol'])).to_csv("../data/valid_set.csv", index=0)
print("get set!")
test_set = get_data(test)
train_set = get_data(train)
valid_set = get_data(valid)
print("data prepared!")

class YModel(nn.Module):
    def __init__(self):
        super(YModel, self).__init__()

        self.lstm = nn.LSTM(input_size=20, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(128*2, 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


def train_model(model):
    print("Training start!")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []

    since = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        model.train()
        for iteration, (train_x, train_y) in enumerate(train_set):
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            train_out = model(train_x)
            train_loss = criterion(train_out, train_y)
            running_loss += train_loss.item()
            train_losses.append(train_loss)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            pred = train_out.argmax(dim=1)
            running_corrects += torch.eq(pred, train_y.argmax(dim=1)).sum()

        print("train loss:{}".format(running_loss / len(train_set)))
        print("train acc:{}".format(running_corrects / len(train_set) / batch))

        running_loss = 0.0
        running_corrects = 0
        model.eval()
        for iteration, (valid_x, valid_y) in enumerate(valid_set):
            valid_x = valid_x.to(device)
            valid_y = valid_y.to(device)
            valid_out = model(valid_x)
            valid_loss = criterion(valid_out, valid_y)
            running_loss += valid_loss.item()
            valid_losses.append(valid_loss)
            pred = valid_out.argmax(dim=1)
            running_corrects += torch.eq(pred, valid_y.argmax(dim=1)).sum()
        print("valid loss:{}".format(running_loss / len(valid_set)))
        print("valid acc:{}".format(running_corrects / len(valid_set) / batch))

        print("epoch:{} {}s\n".format(epoch, (since - time.time()) % 60))
        since = time.time()

    running_loss = 0.0
    running_corrects = 0
    for iteration, (test_x, test_y) in enumerate(test_set):
        test_x = test_x.to(device)
        test_y = test_y.to(device)
        test_out = model(test_x)
        valid_loss = criterion(test_out, test_y)
        running_loss += valid_loss.item()
        valid_losses.append(valid_loss)
        pred = test_out.argmax(dim=1)
        running_corrects += torch.eq(pred, test_y.argmax(dim=1)).sum()
    print("test loss:{}".format(running_loss / len(test_set)))
    print("test acc:{}".format(running_corrects / len(test_set) / batch))

    torch.save(obj=model.state_dict(), f="models/YModel.pth")


def predict(model, x):
    x = get_seq(x)
    model.load_state_dict(torch.load("models/YModel.pth"))
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model.to(device)
    x.to(device)
    out = model(x)
    pred = out.argmax(dim=1)
    if pred == [1, 0]:
        print("不可溶\n")
        return 0
    else:
        print("可溶\n")
        return 1


model = YModel()
train_model(model)
