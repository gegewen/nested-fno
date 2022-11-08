import torch

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        
    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        
# normalization, pointwise gaussian
class TimeGaussianNormalizer(object):
    def __init__(self, x):
        super(TimeGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x,(0,2,3,4,5))[None,:,None,None,None,None]
        self.std = torch.std(x,(0,2,3,4,5))[None,:,None,None,None,None]

    def encode(self, x):
        x = (x - self.mean)/self.std
        return x

    def decode(self, x):
        x = (x * self.std) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        
    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
    
    def param(self):
        return self.mean.reshape(24,), self.std.reshape(24,)