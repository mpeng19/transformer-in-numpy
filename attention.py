'''Multi-Head Attention'''
import numpy as np
from utils import dropout

class SelfAttentionBlock:
    def __init__(self, n_embd, head_size, block_size):
        '''
        one head of self-attention
        Args:
            n_embd: int, the dimension of embedding layer
            head_size: int, the dimension that each head attends to
            block_size: int, the size of the context window
        '''
        
        #q, k, v
        self.key = np.random.randn(hidden_dim, head_size)
        self.query = np.random.randn(hidden_dim, head_size)
        self.value = np.random.randn(hidden_dim, head_size)
        
        #causal mask
        self.causal_mask = np.tril(np.ones((hidden_dim, hidden_dim)))
        
        self.dropout = dropout
        
    
    def forward(self, x):
        """forward pass for the self-attention block"""
        BATCH_SIZE, SEQ_LEN, HIDDEN_DIM = x.shape
        
        #k and q are [batch_size, seq_len, head_size]
        k = self.key @ x
        q = self.query @ x
        
        #compute attention scores
        weights = q @ k.transpose(-2, -1) #[bs, sl, hs] * [bs, hs, sl] = [bs, sl, sl]
        weights = weights / np.sqrt(self.head_size) #scale by sqrt(d_k) to get unit variance
        weights = weights.masked_fill(self.causal_mask == 0, -np.inf) #mask future positions
        
        #softmax
        denom = np.sum(np.exp(weights), axis=-1, keepdims=True)
        num = np.exp(weights)
        probs = num / denom # [bs, sl, sl]
        
        probs = self.dropout(probs)
        
        v = self.value @ x
        output = probs @ v #[bs, sl, sl] * [bs, sl, hs] = [bs, sl, hs]
        return output
    

class MultiHeadAttention:
    def __init__(self, n_embd, n_heads, block_size, head_size):
        '''
        multiple heads of self_attention in parallel
        
        Args:
            n_embd: int, the dimension of embedding layer
            n_heads: int, the number of heads
            block_size: int, the size of the context window
            head_size: int, the dimension that each head attends to
        '''
        self.heads = [SelfAttentionBlock(n_embd, head_size, block_size) for _ in range(n_heads)]
        self.proj = np.random.randn(n_embd, n_embd) #learn to weight each head
        self.dropout = dropout
        
    def forward(self, x):
        out = np.concatenate([head.forward(x) for head in self.heads], axis=-1)
        out = self.dropout(out)
        out = self.proj @ out
        return out
        
