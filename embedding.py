'''Embedding layer for transformer'''
import numpy as np

class EmbeddingLayer:
    def __init__(self, vocab_size, d_model):
        """initialize the embedding layer"""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weights = np.random.randn(vocab_size, d_model)

    def forward(self, x):
        """forward pass for the embedding layer"""
        return self.weights[x]
    

class PositionalEncoding:
    def __init__(self, d_model, max_len=5000):
        """initialize positional encoding"""
        self.d_model = d_model
        self.max_len = max_len
        self.positional_encoding = self.positional_encoding()

    def positional_encoding(self):
        """generate positional encoding"""
        positional_encoding = np.zeros((self.max_len, self.d_model))
        for pos in range(self.max_len):
            for i in range(self.d_model):
                if i % 2 == 0:
                    positional_encoding[pos, i] = np.sin(pos / (10000 ** (i / self.d_model))) #even
                else:
                    positional_encoding[pos, i] = np.cos(pos / (10000 ** ((i - 1) / self.d_model))) #odd
        return positional_encoding

    def forward(self, x):
        """forward pass for the positional encoding"""
        return self.positional_encoding[x]