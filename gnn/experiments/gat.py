from ..models.gat import GAT
from jax import numpy as jnp
import jax

node_feats = jnp.arange(8, dtype=jnp.float32).reshape((1, 4, 2))
adj_matrix = jnp.array(
    [[[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]]
).astype(jnp.float32)

n_features = 2
model = GAT(n_heads_per_layer=(2, 2), c_out=2)
# We define our own parameters here instead of using random initialization
params = model.init_per_layer(
    {
        "projection": {
            "kernel": jnp.eye(n_features),
            "bias": jnp.zeros(n_features),
        },
        "a": jnp.array([[-0.2, 0.3], [0.1, -0.1]]),
    }
)
out_feats = model.apply(params, node_feats, adj_matrix)

print(out_feats)
