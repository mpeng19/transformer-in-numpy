'''
NumPy implementation of AdamW.
'''

import numpy as np
from typing import List, Tuple


class AdamW:
    """Decoupled weight‑decay Adam with optional AMSGrad/maximize flags."""
    def __init__(
        self,
        params: List[np.ndarray],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize

        self.t = 0
        self.m = [np.zeros_like(p) for p in params]  #1st moment
        self.v = [np.zeros_like(p) for p in params]  #2nd moment
        if self.amsgrad:
            self.v_hat_max = [np.zeros_like(p) for p in params]

    def step(self, grads: List[np.ndarray]) -> None:
        """Apply one optimization step *in‑place* to all parameters."""
        self.t += 1
        lr, b1, b2 = self.lr, self.beta1, self.beta2
        t = self.t

        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue #frozen params

            #gradient direction
            if self.maximize:
                g = -g

            #decoupled weight decay
            if self.weight_decay != 0.0:
                p -= lr * self.weight_decay * p

            #running moments
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g * g)

            #bias‑corrected moments
            m_hat = self.m[i] / (1 - b1 ** t)
            if self.amsgrad:
                self.v_hat_max[i] = np.maximum(self.v_hat_max[i], self.v[i])
                v_hat = self.v_hat_max[i] / (1 - b2 ** t)
            else:
                v_hat = self.v[i] / (1 - b2 ** t)

            #parameter update
            p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
