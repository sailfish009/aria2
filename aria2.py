import torch
from torch import nn

class Aria2(nn.Module):
    """
    https://arxiv.org/pdf/1805.08878.pdf
    ARiA2 activation function, a special case of ARiA, for ARiA = f(x, 1, 0, 1, 1, b, 1/a)
    """

    def __init__(self, a=1.5, b = 2.):
        super(Aria2, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x* ((1 + torch.exp(-x)**self.b)**(-self.a))
    

class Gompertz(nn.Module):
    """
    https://forums.fast.ai/t/implementing-new-activation-functions-in-fastai-library/17697
    """

    def __init__(self, a=1., b = 0.5, c=0.5):
        super(Gompertz, self).__init__()
        self.a = b
        self.b = b
        self.c = c

    def forward(self, x):
        gompertz = self.a * torch.exp(-self.b * torch.exp(-self.c * x))
        return gompertz
    

class Aria2_Gompertz(nn.Module):
    """
    test: Aria2 + Gompertz
    """
    
    def __init__(self, a=1., b = 0.5, c=0.5, d=1.5, e=2.):
        super(Aria2_Gompertz, self).__init__()
        self.a = b
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    def forward(self, x):
        gompertz = self.a * torch.exp(-self.b * torch.exp(-self.c * x))
        aria2 = x* ((1 + torch.exp(-x)**self.d)**(-self.e))
        return gompertz * aria2    
