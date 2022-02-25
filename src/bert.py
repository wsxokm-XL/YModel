import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


class YModel(nn.Module):
    def __init__(self):
        super(YModel, self).__init__()

        self.lstm = nn.LSTM(input_size=360, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(128*2, 2)

        self.w = nn.Parameter(torch.Tensor(128*2, 128*2))
        self.u = nn.Parameter(torch.Tensor(128*2, 1))

        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.u, -0.1, 0.1)

    def attention(self, x):
        u = torch.tanh(torch.matmul(x, self.w))
        att = torch.matmul(u, self.u)
        att_score = F.softmax(att, dim=1)

        score_x = x * att_score

        context = torch.sum(score_x, dim=1)
        return context

    def forward(self, x):
        x, _ = self.lstm(x)
        # x = self.fc(x[:, -1, :])
        x = self.attention(x)
        x = self.fc(x)
        return x


batch = 50
lr = 1e-3
num_epochs = 100

net = YModel()

x_train = []
y_train = []
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
train_set = TensorDataset(x_train, y_train)
train_set = DataLoader(train_set, batch_size=batch, shuffle=True)
valid_set = train_set
test_set = train_set


def train_model(model):
    best_acc = 0
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
            running_corrects += torch.eq(pred, train_y).sum()
        print("train loss:{}".format(running_loss / len(train_set)))
        print("train acc:{}".format(running_corrects / len(train_set)))

        running_loss = 0.0
        running_corrects = 0
        model.valid()
        for iteration, (valid_x, valid_y) in enumerate(valid_set):
            valid_x = valid_x.to(device)
            valid_y = valid_y.to(device)
            valid_out = model(valid_x)
            valid_loss = criterion(valid_out, valid_y)
            running_loss += valid_loss.item()
            valid_losses.append(valid_loss)
            pred = valid_out.argmax(dim=1)
            running_corrects += torch.eq(pred, valid_y).sum()
        print("valid loss:{}".format(running_loss / len(valid_set)))
        print("valid acc:{}".format(running_corrects / len(valid_set)))

        print("epoch:{} {}s\n".format(epoch, (since-time.time()) % 60))
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
        running_corrects += torch.eq(pred, test_y).sum()
    print("test loss:{}".format(running_loss / len(test_set)))
    print("test acc:{}".format(running_corrects / len(test_set)))

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

