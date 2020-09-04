import torch

class Aria2(nn.Module):
    """
    ARiA2 activation function, a special case of ARiA, for ARiA = f(x, 1, 0, 1, 1, b, 1/a)
    """

    def __init__(self, a=1.5, b = 2.):
        super(Aria2, self).__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x* ((1 + torch.exp(-x)**self.b)**(-self.a))
