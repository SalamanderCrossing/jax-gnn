import jax.numpy as jnp
import flax.linen as nn


class GATLayer(nn.Module):
    c_out: int  # Dimensionality of output features
    num_heads: int  # Number of heads, i.e. attention mechanisms to apply in parallel.
    concat_heads: bool = True  # If True, the output of the different heads is concatenated instead of averaged.
    alpha: float = 0.2  # Negative slope of the LeakyReLU activation.

    def setup(self):
        if self.concat_heads:
            assert (
                self.c_out % self.num_heads == 0
            ), "Number of output features must be a multiple of the count of heads."
            c_out_per_head = self.c_out // self.num_heads
        else:
            c_out_per_head = self.c_out

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Dense(
            c_out_per_head * self.num_heads,
            kernel_init=nn.initializers.glorot_uniform(),
        )
        self.a = self.param(
            "a", nn.initializers.glorot_uniform(), (self.num_heads, 2 * c_out_per_head)
        )  # One per head

    def __call__(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.shape[0], node_feats.shape[1]

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.reshape((batch_size, num_nodes, self.num_heads, -1))

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # In order to take advantage of JAX's just-in-time compilation, we should not use
        # arrays with shapes that depend on e.g. the number of edges. Hence, we calculate
        # the logit for every possible combination of nodes. For efficiency, we can split
        # a[Wh_i||Wh_j] = a_:d/2 * Wh_i + a_d/2: * Wh_j.
        logit_parent = (node_feats * self.a[None, None, :, : self.a.shape[0] // 2]).sum(
            axis=-1
        )
        logit_child = (node_feats * self.a[None, None, :, self.a.shape[0] // 2 :]).sum(
            axis=-1
        )
        attn_logits = logit_parent[:, :, None, :] + logit_child[:, None, :, :]
        attn_logits = nn.leaky_relu(attn_logits, self.alpha)

        # Mask out nodes that do not have an edge between them
        attn_logits = jnp.where(
            adj_matrix[..., None] == 1.0,
            attn_logits,
            jnp.ones_like(attn_logits) * (-9e15),
        )

        # Weighted average of attention
        attn_probs = nn.softmax(attn_logits, axis=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.transpose(0, 3, 1, 2))
        node_feats = jnp.einsum("bijh,bjhc->bihc", attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(axis=2)

        return node_feats, adj_matrix
