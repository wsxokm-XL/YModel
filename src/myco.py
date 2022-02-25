import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# s = 'MSVHKLTDLRDNSTNWKINVKILSIWNHPPNSHGEITTMILHDDKNNRVDATIPQGNYHNPFCPFLKPGTWIHISDFRVVVPQSRVRYSSFRFHIKFIWETSVHPLPELVKRDFFDIPFDYIVEKTVSTGVLVDVIGALLEVGNLTEDYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTGPQNVIALVSWRLSGIYKLKTLSNDPISKVIANPDIPEVEEIRVVVY '
# d = {}
# for i in s:
#     if i in d:
#         d[i] += 1
#     else:
#         d[i] = 1
#
# print(list(d.keys()))

# a = [1, 2, 5]
# print(type(a))

# dic = {'X': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'A': (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'C': (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'D': (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'E': (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'F': (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'G': (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'H': (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'I': (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'K': (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'L': (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'M': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
#        'N': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
#        'P': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
#        'Q': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
#        'R': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
#        'S': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
#        'T': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
#        'V': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
#        'W': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
#        'Y': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)}
#
#
# data = np.array([['XIPFDYIVEKTVSTGVLVDVIGALLEVGNLTED', 10],
#                 ['YRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLTYRGLKLPFKIMDQYKEVLQCEAHNDQALEFQRFFQRLT', 20],
#                 ['GPQNVIALVSWRLSGIYKLKTLSN', 30]])
#
# ds = np.zeros((1, 360, 20))
# for i in data[:, 0]:
#     dds = np.zeros((1, 20))
#     for j in i:
#         # print(j)
#         # print(dic[j])
#         dds = np.insert(dds, len(dds), dic[j], axis=0)
#     dds = dds[1:dds.shape[0]]
#     print(dds.shape)
#
#     if dds.shape[0] <= 360:
#         dds = np.pad(dds, ((0, 360-dds.shape[0]), (0, 0)), 'constant', constant_values=(0, 0))
#     else:
#         dds = dds[0:360]
#
#     print(dds.shape)
#     ds = np.insert(ds, len(ds), dds, axis=0)
#
# ds = ds[1:ds.shape[0]]
# print(ds.shape)
# print(data.shape)
#
# y = np.array([1, 0, 1])
#
# dx = torch.tensor(ds)
# dy = torch.tensor(y)
# train = TensorDataset(dx, dy)
# # train = DataLoader(train, batch_size= , shuffle=True)
# print(type(train))
# print(dx.shape)
# print(dy)

inputs = [torch.autograd.Variable(torch.randn((2, 3))) for _ in range(5)]
print(inputs)
inputs = np.array([[1, 2], [2, 3], [3, 4]])
inputs = torch.tensor(inputs)
for i in range(inputs.shape[0]):
    # print(i)
    print(inputs[i])

# print(inputs[2])
