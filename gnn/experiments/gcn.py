from ..models.gcn import GCN
from jax import numpy as jnp

node_feats = jnp.arange(8, dtype=jnp.float32).reshape((1, 4, 2))
adj_matrix = jnp.array(
    [[[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]]
).astype(jnp.float32)


model = GCN(n_layers=2, n_features=2)

# We define our own parameters here instead of using random initialization
params = {
    "projection": {
        "kernel": jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        "bias": jnp.array([0.0, 0.0]),
    }
}
out_feats = model.apply({"params": params}, node_feats, adj_matrix)
print(out_feats)
