'''activations and other utilities'''
import numpy as np

def relu(x):
    x[x<0] = 0
    return x

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def dropout(x, p=0.1):
    if p > 0:
        mask = np.random.binomial(1, 1-p, x.shape)
        return x * mask
    return x

def cross_entropy(logits, targets):
    return -np.sum(targets * np.log(logits)) / logits.shape[0] #divide by batch size to get average loss


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        '''
        implements layer norm - https://arxiv.org/abs/1607.06450
        
        Args:
            dim: int, the dimension of the input
            eps: float, a small constant to avoid division by zero
        '''
        self.eps = eps
        self.g = np.ones(dim)
        self.b = np.zeros(dim)
        
    def __call__(self, x):
        xmean = np.mean(x, axis=-1, keepdims=True)
        xvar = np.var(x, axis=-1, keepdims=True)
        xnorm = (x - xmean) / np.sqrt(xvar + self.eps)
        return self.g * xnorm + self.b
    
    def backward(self, d_out):
        xmean = np.mean(x, axis=-1, keepdims=True)
        xvar = np.var(x, axis=-1, keepdims=True)
        xnorm = (x - xmean) / np.sqrt(xvar + self.eps)
        return d_out * self.g * (1 - xnorm**2) / np.sqrt(xvar + self.eps)
    
    def parameters(self):
        return [self.g, self.b]