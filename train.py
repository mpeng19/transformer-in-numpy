'''Training loop'''

import argparse
import time
from typing import Dict, List

import numpy as np

from dataloader import FineWebDataLoader
from langugage_model import DecoderOnlyTransformer
from backprop import backward_pass
from optimizer import AdamW

#config
SEQUENCE_LENGTH = 128
BATCH_SIZE = 8
NUM_EXAMPLES = None
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 1
LR = 1e-4
WEIGHT_DECAY = 1e-2
AMSGRAD = False
EPOCHS = 1
STEPS = None
LOG_EVERY = 10

def build_param_lookup(model) -> Dict[str, np.ndarray]:
    """Return **parameter name -> np.ndarray** mapping following the naming convention used inside `backward_pass`."""
    params: Dict[str, np.ndarray] = {}

    params["tok_weights"] = model.token_embedding.weights

    #transformer blocks
    for idx, blk in enumerate(model.blocks):
        mha = blk.attn if hasattr(blk, "attn") else blk
        params[f"block{idx}_proj"] = mha.proj
        for h_i, head in enumerate(mha.heads):
            params[f"block{idx}_query_{h_i}"] = head.query
            params[f"block{idx}_key_{h_i}"] = head.key
            params[f"block{idx}_value_{h_i}"] = head.value

    #LayerNorm and LM head
    params["ln_f_g"] = model.ln_f.g
    params["ln_f_b"] = model.ln_f.b
    params["lm_head"] = model.lm_head
    return params


def make_ordered_param_list(param_lookup: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Match the alphabetical order used for the grad list coming out of`backward_pass`."""
    return [param_lookup[k] for k in sorted(param_lookup.keys())]

def main():
    #Data
    loader = FineWebDataLoader(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        num_examples=NUM_EXAMPLES,
    )
    vocab_size = loader.get_vocab_size()

    #Model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        n_embd=D_MODEL,
        n_heads=N_HEADS,
        block_size=SEQUENCE_LENGTH,
        head_size=D_MODEL // N_HEADS,
        n_layers=N_LAYERS,
    )

    #Optimiser
    param_lookup = build_param_lookup(model)
    param_list = make_ordered_param_list(param_lookup)
    optimiser = AdamW(
        params=param_list,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )

    #Training loop
    global_step = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_tokens = 0
        start = time.time()

        for batch in loader:
            #each batch is (B, T)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            #forward
            logits, _ = model.forward(input_ids, targets=None)
            
            #backward
            flat_logits = logits.reshape(-1, vocab_size)
            flat_targets = target_ids.reshape(-1)
            loss, grads = backward_pass(model, flat_logits, flat_targets)

            #update
            optimiser.step(grads)

            #log loss
            epoch_loss += loss * input_ids.shape[0]
            epoch_tokens += input_ids.shape[0]
            global_step += 1

            if global_step % LOG_EVERY == 0:
                print(
                    f"step {global_step:6d} | epoch {epoch:2d} | loss {loss:8.4f} | "
                    f"tokens/sec {epoch_tokens / (time.time() - start):.1f}")

            if STEPS and global_step >= STEPS:
                break
        if STEPS and global_step >= STEPS:
            break

        mean_loss = epoch_loss / epoch_tokens
        print(f"Epoch {epoch} finished – mean loss: {mean_loss:.4f}")


if __name__ == "__main__":
    main()
