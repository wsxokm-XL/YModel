# class Protein(object):
#     def __init__(self, name, exp, sol, seq):
#         self.name = name
#         self.exp = exp
#         self.sol = sol
#         self.seq = seq
#
#
# pro = [Protein(1, 2, 3, 4), Protein(1, 2, 3, 4)]
# print(pro[0].exp)
import torch
print(torch.cuda.is_available())
