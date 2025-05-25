'''
Backward‑pass gradients for each module.

accumulates gradients on ```self.grads``` ( a dict mapping parameter names to dθ).

Only a *single* transformer block is supported for clarity, but the logic can be extended to an arbitrary depth.

The entry‑point ``backward_pass(model, logits, targets)``
    1. builds ``d_logits`` from the cross‑entropy loss,
    2. walks the model **in reverse topological order**,
    3. fills ``grads`` attributes on every leaf module.

Finally, it returns ``loss`` and a **flat list** ``all_grads`` that aligns
one‑to‑one (in order) with ``model.parameters()``` in PyTorch.
'''

import numpy as np
from typing import Tuple, List

def softmax_backward(d_out: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Vectorised jacobian‑vector product for softmax.

    Given dL/dy (``d_out``) where y = softmax(z) and the already computed
    y = ``probs``, returns dL/dz.
    """
    s = probs
    return (d_out - np.sum(d_out * s, axis=-1, keepdims=True)) * s


def cross_entropy_backward(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """dL/dlogits for *integer* targets (shape (N,))."""
    N, C = logits.shape
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    d_logits = probs.copy()
    d_logits[np.arange(N), targets] -= 1
    d_logits /= N  #average over batch
    return d_logits

class EmbeddingLayerGradMixin:
    """Adds a ``backward`` method to the existing EmbeddingLayer class."""
    def forward(self, idx: np.ndarray):
        out = super().forward(idx)
        self.cache = (idx,)
        return out

    def backward(self, d_out: np.ndarray):
        idx, = self.cache
        self.grads = {
            "weights": np.zeros_like(self.weights)
        }
        flat_idx = idx.reshape(-1)
        flat_grad = d_out.reshape(-1, self.d_model)
        np.add.at(self.grads["weights"], flat_idx, flat_grad)
        #embedding layer has no input gradient (idx is int)
        return None


class LayerNormGradMixin:
    """Backward pass for utils.LayerNorm (affine variant)."""
    def __call__(self, x): 
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        out = self.g * self.x_hat + self.b
        return out

    def backward(self, d_out: np.ndarray):
        N = self.x.shape[-1]
        dbeta = d_out.sum(axis=(0, 1))  #sum over batch & time
        dgamma = np.sum(d_out * self.x_hat, axis=(0, 1))

        dx_hat = d_out * self.g
        dvar = np.sum(dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** -1.5, axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True) + dvar * np.mean(-2 * (self.x - self.mean), axis=-1, keepdims=True)
        dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.mean) / N + dmean / N

        self.grads = {"g": dgamma, "b": dbeta}
        return dx


class SelfAttentionBlockGradMixin:
    """Implements backward pass for attention head."""
    def forward(self, x):
        B, T, _ = x.shape
        
        q = x @ self.query
        k = x @ self.key
        v = x @ self.value
        
        att = (q @ k.swapaxes(1, 2)) / np.sqrt(self.head_size)
        att = np.where(self.causal_mask[:T, :T] == 1, att, -1e9)
        probs = np.exp(att - att.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        out = probs @ v
        self.cache = (x, q, k, v, probs)
        return out

    def backward(self, d_out: np.ndarray):
        x, q, k, v, probs = self.cache
        B, T, hs = q.shape

        d_probs = d_out @ v.swapaxes(1, 2) #(B,T,T)
        d_v = probs.swapaxes(1, 2) @ d_out #(B,T,hs)

        d_att = softmax_backward(d_probs, probs) #(B,T,T)

        scale = 1.0 / np.sqrt(hs)
        d_q = d_att @ k * scale #(B,T,hs)
        d_k = d_att.swapaxes(1, 2) @ q * scale #(B,T,hs)

        d_query = x.reshape(-1, x.shape[-1]).T @ d_q.reshape(-1, hs)
        d_key = x.reshape(-1, x.shape[-1]).T @ d_k.reshape(-1, hs)
        d_value = x.reshape(-1, x.shape[-1]).T @ d_v.reshape(-1, hs)

        d_x_q = d_q @ self.query.T
        d_x_k = d_k @ self.key.T
        d_x_v = d_v @ self.value.T
        d_x = d_x_q + d_x_k + d_x_v #(B,T,H)

        self.grads = {
            "query": d_query,
            "key":   d_key,
            "value": d_value,
        }
        return d_x


class MultiHeadAttentionGradMixin:
    """Merge grads from each head and final projection."""
    def backward(self, d_out: np.ndarray):
        out_cat = self.out_cat  #saved during forward
        d_proj = out_cat.reshape(-1, out_cat.shape[-1]).T @ d_out.reshape(-1, d_out.shape[-1])
        d_out_cat = d_out @ self.proj.T

        #accumulate input grad
        splits = np.split(d_out_cat, len(self.heads), axis=-1)
        d_x_total = 0
        for head, d_split in zip(self.heads, splits):
            d_x_total += head.backward(d_split)

        self.grads = {"proj": d_proj}
        return d_x_total

    # monkey‑patch *forward* so we stash concatenated output
    def forward(self, x):
        out_cat = np.concatenate([h.forward(x) for h in self.heads], axis=-1)
        self.out_cat = out_cat
        y = out_cat @ self.proj
        return y

def backward_pass(model, logits: np.ndarray, targets: np.ndarray) -> Tuple[float, List[np.ndarray]]:
    '''
    Runs a single backward pass.

    Args:
        model: DecoderOnlyTransformer (already executed forward!)
        logits: np.ndarray
            Raw predictions returned by the model (shape (B*T, vocab)).
        targets: np.ndarray
            Ground‑truth token ids flattened to shape (B*T,).
    Returns:
        loss: float
    '''
    #d loss / d logits
    d_logits = cross_entropy_backward(logits, targets)

    N = logits.shape[0]
    log_probs = logits - logits.max(axis=1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(N), targets])

    #lm_head
    x_last = model.x_last
    d_lm_head = x_last.reshape(-1, x_last.shape[-1]).T @ d_logits  #(n_embd,vocab)
    d_x_last = d_logits @ model.lm_head.T

    #accumulate
    param_grads = {"lm_head": d_lm_head}

    #layer norm
    d_x = model.ln_f.backward(d_x_last)
    param_grads.update({f"ln_f_{k}": v for k, v in model.ln_f.grads.items()})

    #transformer block(s) (reverse order)
    for blk_idx in reversed(range(len(model.blocks))):
        blk = model.blocks[blk_idx]
        d_x = blk.backward(d_x)
        for name, g in blk.grads.items():
            param_grads[f"block{blk_idx}_{name}"] = g

    #embeddings
    d_tok_emb = d_x
    model.token_embedding.backward(d_tok_emb)
    param_grads.update({f"tok_{k}": v for k, v in model.token_embedding.grads.items()})

    #flatten grads in deterministic order
    all_grads = [param_grads[k] for k in sorted(param_grads.keys())]
    return loss, all_grads
