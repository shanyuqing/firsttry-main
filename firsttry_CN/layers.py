import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
    
        # 确保两个张量的数据类型一致
        if isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj).float()  # 如果 adj 是 numpy.ndarray，则转换为 tensor
        support = support.to(torch.float32)  # 确保 support 张量也是 float32
        output = torch.mm(adj, support)

        if self.bias is not None:
            return (output + self.bias).to(device)
        else:
            return output.to(device)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
