import numpy
import torch
from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from d2l import torch as d2l

batch = 2

x_train = torch.randn(5, 4, 2)
y_train = torch.randn(5, 1, 2)

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
print(len(train_dl))
for iteration, (train_x, train_y) in enumerate(train_dl):
    print(iteration, train_x, train_y)
    print(len(train_dl)*len(train_y))
# def test():
#     if torch.cuda.is_available():
#         device = torch.device('cuda:0')
#     else:
#         device = torch.device('cpu')
#
#     batch_size, num_steps = 32, 35
#     train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
#     vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
#     num_inputs = vocab_size
#     lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
#     model = d2l.RNNModel(lstm_layer, len(vocab))
#     model = model.to(device)
#
#     num_epochs, lr = 500, 1
#     d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
