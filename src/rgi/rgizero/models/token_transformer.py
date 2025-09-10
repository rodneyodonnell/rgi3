"""
Model definition, based on https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from rgi.rgizero.models.transformer import Transformer, LayerNorm, init_weights, configure_optimizers


class TokenTransformer(nn.Module):
    """Transformer model with tokens as inputs and outputs."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.wte = nn.Embedding(self.vocab_size, config.n_embd)  # token embeddings
        self.transformer = Transformer(config)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, self.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.reset_parameters()

    def reset_parameters(self):
        # init all weights in this module tree
        self.apply(init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx, targets=None):
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer(tok_emb)
        x = self.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return configure_optimizers(self, weight_decay, learning_rate, betas, device_type)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at n_max_context
            ctx = self.config.n_max_context if hasattr(self.config, "n_max_context") else self.config.block_size
            idx_cond = idx if idx.size(1) <= ctx else idx[:, -ctx:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
