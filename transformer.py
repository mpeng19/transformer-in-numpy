'''Implements the transformer architecture'''
import numpy as np
from attention import MultiHeadAttention
from embedding import EmbeddingLayer, PositionalEncoding
from utils import relu, softmax, dropout, LayerNorm

class TransformerBlock:
    def __init__(self, n_embd, n_heads, block_size, head_size):
        '''
        initializes a transformer block from the attention paper: https://arxiv.org/abs/1706.03762
        Args:
            n_embd: int, the dimension of embedding layer
            n_heads: int, the number of attention heads
            block_size: int, the size of the context window
            head_size: int, the dimension that each head attends to
        '''
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_embd, n_heads, block_size, head_size)
        self.ff = FeedFowardNetwork(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        
    def forward(self, x):
        """forward pass for the transformer block"""
        #add x to the output of the self-attention layer - residual connections: https://arxiv.org/abs/1512.03385
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class FeedFowardNetwork:
    def __init__(self, n_embd):
        '''
        single layer followed by a non-linearity
        Args:
            n_embd: int, the dimension of embedding layer
            n_hidden: int, the dimension of the hidden layer
        '''
        self.w1 = np.random.randn(n_embd, 4 * n_embd)
        self.a1 = relu
        self.w2 = np.random.randn(4 * n_embd, n_embd)
        self.dropout = dropout
        
    def forward(self, x):
        """forward pass for the feedforward network"""
        z1 = x @ self.w1
        a1 = self.a1(z1)
        z2 = a1 @ self.w2
        return self.dropout(z2)