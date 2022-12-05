from flax import linen as nn
from jax import numpy as jnp
from .gat_layer import GATLayer
from typing import Any

# GCN is a sequential model of GCN layers
class GAT(nn.Module):
    n_heads_per_layer: tuple[int, ...]
    c_out: int

    def init_per_layer(self, params_per_layer: dict[str, Any]):
        return {
            "params": {
                f"GATLayer_{i}": params_per_layer
                for i in range(len(self.n_heads_per_layer))
            },
        }

    @nn.compact
    def __call__(self, node_features: jnp.ndarray, adj: jnp.ndarray):
        layers = [
            GATLayer(self.c_out, self.n_heads_per_layer[i])
            for i in range(len(self.n_heads_per_layer))
        ]
        for layer in layers:
            node_features, adj = layer(node_features, adj)
        return node_features
