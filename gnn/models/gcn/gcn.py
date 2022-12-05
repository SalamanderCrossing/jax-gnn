from flax import linen as nn
from jax import numpy as jnp
from .gcn_layer import GCNLayer

# GCN is a sequential model of GCN layers
class GCN(nn.Module):
    n_layers: int
    n_features: int

    @nn.compact
    def __call__(self, node_features: jnp.ndarray, adj: jnp.ndarray):
        return nn.Sequential(
            [GCNLayer(self.n_features) for _ in range(self.n_layers)]
        )(node_features, adj)
