"""Three-layer, single-head decoder used for the four-step experiment."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width: int, ffn_width: int, normalization: str = "block_end"):
        super().__init__()
        self.normalization = normalization
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.ReLU(),
                                 nn.Linear(ffn_width, width))
        # No affine parameter: RMSNorm's per-coordinate gain is fixed at 1.
        if normalization == "prelayernorm":
            self.norm_attention = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
            self.norm_ffn = nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
        elif normalization == "prenorm":
            self.norm_attention = nn.RMSNorm(width, eps=1e-6, elementwise_affine=False)
            self.norm_ffn = nn.RMSNorm(width, eps=1e-6, elementwise_affine=False)
        else:
            self.norm = nn.RMSNorm(width, eps=1e-6, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        attention_input = self.norm_attention(x) if self.normalization in ("prenorm", "prelayernorm") else x
        q, k, v = self.qkv(attention_input).view(batch, length, 3, width).unbind(2)
        attended = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
        ).squeeze(1)
        after_attention = x + self.out(attended)
        if self.normalization in ("prenorm", "prelayernorm"):
            return after_attention + self.ffn(self.norm_ffn(after_attention))
        return self.norm(after_attention + self.ffn(after_attention))


class ReasoningTransformer(nn.Module):
    def __init__(self, width: int = 1024, ffn_width: int = 2048,
                 layers: int = 3, vocab: int = 120, length: int = 31,
                 initialization: str = "kaiming_uniform_relu_gamma1", normalization: str = "prenorm"):
        super().__init__()
        if initialization not in ("uniform_gamma1", "kaiming_uniform_relu", "kaiming_uniform_relu_gamma1"):
            raise ValueError(f"Unknown initialization: {initialization}")
        if normalization not in ("block_end", "prenorm", "prelayernorm"):
            raise ValueError(f"Unknown normalization: {normalization}")
        self.normalization = normalization
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Embedding(length, width)
        self.blocks = nn.ModuleList(Block(width, ffn_width, normalization) for _ in range(layers))
        self.final_norm = (nn.RMSNorm(width, eps=1e-6, elementwise_affine=False)
                           if normalization == "prenorm" else
                           nn.LayerNorm(width, eps=1e-6, elementwise_affine=False)
                           if normalization == "prelayernorm" else nn.Identity())
        self.head = nn.Linear(width, vocab)
        # Word and position embeddings use N(0, 1). The default linear
        # initialization is gamma=1; the optional comparison uses Kaiming
        # uniform fan_in/ReLU weights and standard fan_in uniform biases.
        # RMSNorm has no trainable gain.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if initialization in ("kaiming_uniform_relu", "kaiming_uniform_relu_gamma1"):
                    nn.init.kaiming_uniform_(module.weight, mode="fan_in",
                                            nonlinearity="relu")
                    bound = 1.0 / math.sqrt(module.in_features)
                    if initialization == "kaiming_uniform_relu_gamma1":
                        # Retain Kaiming's ReLU gain, but use fan_in**(-1)
                        # instead of fan_in**(-1/2) for weights and biases.
                        with torch.no_grad():
                            module.weight.div_(math.sqrt(module.in_features))
                        bound = 1.0 / module.in_features
                else:
                    bound = math.sqrt(3.0) / module.in_features
                    nn.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -bound, bound)
        nn.init.normal_(self.token.weight, mean=0, std=1)
        nn.init.normal_(self.position.weight, mean=0, std=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token(tokens) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x[:, -1]))
