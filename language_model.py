'''implements a decoder only transformer architecture for language modeling'''
import numpy as np
from transformer import TransformerBlock
from embedding import EmbeddingLayer, PositionalEncoding
from utils import relu, softmax, dropout, cross_entropy, LayerNorm

class DecoderOnlyTransformer:
    def __init__(self, vocab_size, n_embd, n_heads, block_size, head_size, n_layers, dropout=0.1):
        '''
        initializes a decoder only transformer architecture for language modeling
        
        Args:
            vocab_size: int, the size of the vocabulary
            n_embd: int, the dimension of embedding layer
            n_heads: int, the number of attention heads
            block_size: int, the size of the context window
            head_size: int, the dimension that each head attends to
            n_layers: int, the number of transformer blocks
            dropout: float, the dropout rate
        '''
        self.token_embedding = EmbeddingLayer(vocab_size, n_embd)
        self.positional_encoding = PositionalEncoding(n_embd, block_size)
        self.blocks = [TransformerBlock(n_embd, n_heads, block_size, head_size) for _ in range(n_layers)]#we are going to use only 1 block to make calculating gradients easier
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = np.random.randn(n_embd, vocab_size)
        
    def forward(self, ids, targets=None):
        """forward pass for the decoder only transformer"""
        B, T = idx.shape
        tok_embd = self.token_embedding(idx)
        pos_embd = self.positional_encoding(tok_embd)
        x = tok_embd + pos_embd
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.lm_head.T
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = cross_entropy(logits, targets)
        return logits, loss

    
    #TODO: generate function