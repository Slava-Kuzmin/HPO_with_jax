import math
import string
import jax
import jax.numpy as jnp
import tensorcircuit as tc
from flax import linen as nn

tc.set_backend("jax")

# ===============================================================
# Fourier Neural Network (FNN) using Flax and JAX
# ===============================================================
class FNN(nn.Module):
    num_features: int = 8            # number of input features
    num_frequencies: int = 1         # number of sin/cos frequencies per feature
    num_output: int = 1              # output dimension
    init_std: float = 0.1            # stddev for weight init
    frequency_min_init: float = 1.0  # initial scale for input frequencies
    trainable_frequency_min: bool = True

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]

        # --- 1) Frequency scaling parameter ---
        if self.trainable_frequency_min:
            frequency_min = self.param(
                "frequency_min",
                lambda rng: jnp.array(self.frequency_min_init, dtype=x.dtype),
            )
        else:
            frequency_min = jnp.array(self.frequency_min_init, dtype=x.dtype)
        x_scaled = x * frequency_min  # shape: (B, num_features)

        # --- 2) Build Fourier features ---
        # physical feature dimension per input: 1 + 2*num_frequencies
        D_phys = 1 + 2 * self.num_frequencies

        # prepare frequencies [1, 2, ..., num_frequencies]
        frequencies = jnp.arange(1, self.num_frequencies + 1, dtype=x.dtype)

        # compute sin and cos features: shapes (B, num_features, num_frequencies)
        sin_feats = jnp.sin(x_scaled[..., None] * frequencies)
        cos_feats = jnp.cos(x_scaled[..., None] * frequencies)

        # constant '1' channel: shape (B, num_features, 1)
        ones = jnp.ones((batch_size, self.num_features, 1), dtype=x.dtype)

        # concatenate into (B, num_features, D_phys)
        features = jnp.concatenate([ones, sin_feats, cos_feats], axis=-1)

        # --- 3) Learnable tensor of shape (D_phys, D_phys, ..., D_phys, num_output) ---
        weight_shape = (D_phys,) * self.num_features + (self.num_output,)
        W = self.param(
            "W",
            nn.initializers.normal(stddev=self.init_std),
            weight_shape,
        )
        bias = self.param("bias", nn.initializers.zeros, (self.num_output,))

        # helper to contract a single sample's features through the high-order tensor
        def contract_single(sample_features):
            # sample_features: (num_features, D_phys)
            tensor = W
            # sequentially contract each feature mode
            for i in range(self.num_features):
                tensor = jnp.tensordot(sample_features[i], tensor, axes=([0], [0]))
            # result has shape (num_output,)
            return tensor

        # apply to each element in the batch
        result = jax.vmap(contract_single)(features)  # shape: (B, num_output)

        return result + bias


import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# ===============================================================
# Quantum Circuit Function
# ===============================================================
def quantum_circuit(x, weights, entanglement_weights, final_rotations, num_qubits, layer_depth, num_frequencies):
    """
    Build a quantum circuit using tensorcircuit, with parameterized RXX entanglement and final rotations.

    Args:
      x: JAX array of shape (num_qubits,) representing the input.
      weights: Array of shape (num_frequencies, layer_depth, num_qubits, 3) for single-qubit rotations.
      entanglement_weights: Array of shape (num_frequencies, layer_depth, num_qubits-1) for RXX angles.
      final_rotations: Array of shape (num_qubits, 3) for the final variational rotation layer.
      num_qubits: Number of qubits.
      layer_depth: Number of variational layers per reuploading block.
      num_frequencies: Number of reuploading blocks.

    Returns:
      JAX array of shape (num_qubits,) with the expectation values of PauliZ.
    """
    import tensorcircuit as tc  # assuming tensorcircuit is available
    c = tc.Circuit(num_qubits)
    for f in range(num_frequencies):
        # Input encoding
        for j in range(num_qubits):
            c.rx(j, theta=x[j])
        # Variational layers
        for k in range(layer_depth):
            for j in range(num_qubits):
                theta, phi, alpha = weights[f, k, j]
                c.r(j, theta=theta, phi=phi, alpha=alpha)
            for j in range(num_qubits - 1):
                theta_ent = entanglement_weights[f, k, j]
                c.rxx(j, j + 1, theta=theta_ent)
    # Final rotation layer
    for j in range(num_qubits):
        theta, phi, alpha = final_rotations[j]
        c.r(j, theta=theta, phi=phi, alpha=alpha)
    out = [c.expectation_ps(z=[j]) for j in range(num_qubits)]
    return jnp.real(jnp.array(out))

# ===============================================================
# Quantum Neural Network (QNN) with tensorcircuit
# ===============================================================
class QNN(nn.Module):
    num_features: int = 8         # classical input; num_qubits computed automatically
    num_frequencies: int = 1      
    layer_depth: int = 1          
    num_output: int = 1
    init_std: float = 0.1       # used in final dense layer
    init_std_Q: float = 0.1       # used for quantum circuit weights
    frequency_min_init: float = 1.0  
    trainable_frequency_min: bool = True

    @property
    def num_qubits(self):
        # In this design, we assume one qubit per classical feature.
        return self.num_features

    @nn.compact
    def __call__(self, x):
        # x: shape (batch_size, num_qubits)
        num_qubits = self.num_qubits
        # Frequency scaling.
        frequency_min = (
            self.param("frequency_min", lambda rng: jnp.array(self.frequency_min_init, dtype=x.dtype))
            if self.trainable_frequency_min else self.frequency_min_init
        )
        x_scaled = x * frequency_min

        # Define shapes for the combined weights.
        shape_weights = (self.num_frequencies, self.layer_depth, num_qubits, 3)
        shape_entanglement = (self.num_frequencies, self.layer_depth, num_qubits - 1)
        shape_final = (num_qubits, 3)

        # Compute total sizes as Python integers.
        size_weights = int(np.prod(shape_weights))
        size_entanglement = int(np.prod(shape_entanglement))
        size_final = int(np.prod(shape_final))
        total_size = size_weights + size_entanglement + size_final

        # Combined parameter for all quantum circuit weights.
        quanum_weights = self.param(
            "quanum_weights",
            lambda rng, shape: jax.random.normal(rng, shape) * self.init_std_Q,
            (total_size,)
        )

        # Slice and reshape to recover individual parameters.
        weights = jnp.reshape(quanum_weights[:size_weights], shape_weights)
        entanglement_weights = jnp.reshape(
            quanum_weights[size_weights:size_weights + size_entanglement], shape_entanglement
        )
        final_rotations = jnp.reshape(
            quanum_weights[size_weights + size_entanglement:], shape_final
        )

        # Build the quantum circuit output (vectorized over the batch).
        circuit_out = jax.vmap(
            lambda single_x: quantum_circuit(
                single_x,
                weights,
                entanglement_weights,
                final_rotations,
                num_qubits,
                self.layer_depth,
                self.num_frequencies
            )
        )(x_scaled)
        # Use a dense layer on the circuit output.
        output = nn.Dense(self.num_output,
                          kernel_init=nn.initializers.normal(stddev=self.init_std)
                         )(circuit_out)
        return output

# ===============================================================
# Amplitude Encoding Neural Network (AENN) with tensorcircuit
# ===============================================================
def amplitude_encoding_circuit(x, quantum_weights, num_qubits, layer_depth):
    """
    Build a quantum circuit using tensorcircuit with amplitude encoding,
    a final rotation layer, and parametrized gates.

    Args:
      x: JAX array of shape (2**num_qubits,) representing the input state.
      quantum_weights: 1D JAX array containing all weights (both rotations and RXX entangling gate parameters).
      num_qubits: Number of qubits.
      layer_depth: Number of variational layers.

    Returns:
      JAX array of shape (2**num_qubits,) with the state probabilities.
    """
    # Determine sizes for the rotations and rxx parameters.
    size_rotations = (layer_depth + 1) * num_qubits * 3
    size_rxx = layer_depth * (num_qubits - 1)
    
    # Split and reshape the parameters.
    rotations = jnp.reshape(quantum_weights[:size_rotations], (layer_depth + 1, num_qubits, 3))
    rxx = jnp.reshape(quantum_weights[size_rotations:], (layer_depth, num_qubits - 1))
    
    c = tc.Circuit(num_qubits, inputs=x)
    
    # Variational layers with rotations and parametrized RXX gates.
    for k in range(layer_depth):
        # Apply single-qubit rotations.
        for j in range(num_qubits):
            theta, phi, alpha = rotations[k, j]
            c.r(j, theta=theta, phi=phi, alpha=alpha)
        # Apply parametrized RXX gates (excluding the connection from last to first).
        if num_qubits > 1:
            for j in range(num_qubits - 1):
                c.rxx(j, j + 1, theta=rxx[k, j])
    
    # Final rotation layer using the last set of rotation parameters.
    for j in range(num_qubits):
        theta, phi, alpha = rotations[layer_depth, j]
        c.r(j, theta=theta, phi=phi, alpha=alpha)
    
    return jnp.abs(c.state()) ** 2




class AENN(nn.Module):
    num_features: int = 4
    layer_depth: int = 1
    num_output: int = 1
    init_std: float = jnp.pi / 8
    init_std_Q: float = jnp.pi / 8

    @property
    def num_qubits(self):
        return math.ceil(math.log2(self.num_features))

    @nn.compact
    def __call__(self, x):
        num_qubits = self.num_qubits
        
        # Normalize input.
        norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        norm = jnp.clip(norm, a_min=1e-8)
        x_norm = x / norm

        # Compute the total number of circuit parameters.
        size_rotations = (self.layer_depth + 1) * num_qubits * 3
        size_rxx = self.layer_depth * (num_qubits - 1)
        total_size = size_rotations + size_rxx

        # Create one single parameter tensor for all quantum weights.
        quantum_weights = self.param(
            "quantum_weights",
            lambda rng: jax.random.normal(rng, (total_size,)) * self.init_std_Q
        )

        # Vectorized evaluation over the input batch.
        circuit_out = jax.vmap(
            lambda single_x: amplitude_encoding_circuit(
                single_x, quantum_weights, num_qubits, self.layer_depth
            )
        )(x_norm)

        # Undo the normalization by multiplying by the squared norm.
        circuit_out = (norm ** 2) * circuit_out

        # Final dense layer to produce the output.
        output = nn.Dense(
            features=self.num_output,
            use_bias=True,
            kernel_init=nn.initializers.normal(stddev=self.init_std)
        )(circuit_out)
        return output




# ===============================================================
# Square Linear Model with Square Activation
# ===============================================================
class SquareLinearModel(nn.Module):
    num_features: int = 8  # input feature dimension for the first layer
    num_output: int = 1      # number of output features
    init_std: float = 0.1

    @nn.compact
    def __call__(self, x):
        # First linear layer without bias.
        x = nn.Dense(features=self.num_features, use_bias=False,
                     kernel_init=nn.initializers.normal(stddev=self.init_std))(x)
        x = jnp.square(x)
        output = nn.Dense(features=self.num_output, use_bias=True,
                          kernel_init=nn.initializers.normal(stddev=self.init_std))(x)
        return output

# ===============================================================
# Matrix Product State Fourier Neural Network (MPS_FNN) in Flax
# ===============================================================
class MPS_FNN(nn.Module):
    num_features: int           # number of input sites (classical features)
    bond_dim: int                      # maximum virtual bond dimension
    num_frequencies: int        # number of frequencies (for feature mapping)
    num_output: int = 1
    init_std: float = 0.1
    frequency_min_init: float = 1.0
    trainable_frequency_min: bool = True

    @nn.compact
    def __call__(self, x):
        # x: shape (B, num_features)
        B = x.shape[0]
        D_phys = 1 + 2 * self.num_frequencies  # physical dimension per site
        frequency_min = (
            self.param("frequency_min", lambda rng: jnp.array(self.frequency_min_init, dtype=x.dtype))
            if self.trainable_frequency_min else self.frequency_min_init
        )
        x_scaled = x * frequency_min

        # Compute bond dimensions.
        bond_dims = []
        for b in range(self.num_features + 1):
            if 0 < b < self.num_features:
                d = min(b, self.num_features - b)
                bond_dims.append(min(self.bond_dim, D_phys ** d))
            else:
                bond_dims.append(1)
        center = (self.num_features // 2 - 1) if (self.num_features % 2 == 0) else (self.num_features // 2)
        ones = jnp.ones((B, self.num_features, 1), dtype=x.dtype)
        x_expanded = x_scaled[..., None]
        freqs = jnp.arange(1, self.num_frequencies + 1, dtype=x.dtype)
        sin_vals = jnp.sin(x_expanded * freqs)
        cos_vals = jnp.cos(x_expanded * freqs)
        features = jnp.concatenate([ones, sin_vals, cos_vals], axis=-1)

        log_norm = jnp.zeros((B,), dtype=x.dtype)
        # --- Left block contraction ---
        left_state = jnp.ones((B, bond_dims[0]), dtype=x.dtype)
        for i in range(center):
            A = self.param(f"mps_{i}", nn.initializers.normal(stddev=self.init_std),
                           (bond_dims[i], D_phys, bond_dims[i+1]))
            v = features[:, i, :]
            left_state = jnp.einsum("bi,ipj,bp->bj", left_state, A, v)
            norm_val = jnp.linalg.norm(left_state, axis=1, keepdims=True) + 1e-8
            left_state = left_state / norm_val
            log_norm = log_norm + jnp.log(norm_val.squeeze(axis=1))
        
        # --- Right block contraction ---
        right_state = jnp.ones((B, bond_dims[-1]), dtype=x.dtype)
        for i in range(self.num_features - 1, center, -1):
            A = self.param(f"mps_{i}", nn.initializers.normal(stddev=self.init_std),
                           (bond_dims[i], D_phys, bond_dims[i+1]))
            v = features[:, i, :]
            right_state = jnp.einsum("lpr,bp,br->bl", A, v, right_state)
            norm_val = jnp.linalg.norm(right_state, axis=1, keepdims=True) + 1e-8
            right_state = right_state / norm_val
            log_norm = log_norm + jnp.log(norm_val.squeeze(axis=1))
        
        # --- Center tensor contraction ---
        A_center = self.param(f"mps_{center}", nn.initializers.normal(stddev=self.init_std),
                              (bond_dims[center], D_phys, bond_dims[center+1], self.num_output))
        v_center = features[:, center, :]
        temp = jnp.einsum("bi,iprn,bp->brn", left_state, A_center, v_center)
        output = jnp.einsum("brn,br->bn", temp, right_state)
        norm_val = jnp.linalg.norm(output, axis=1, keepdims=True) + 1e-8
        output = output / norm_val
        log_norm = log_norm + jnp.log(norm_val.squeeze(axis=1))
        
        return output * jnp.exp(log_norm)[..., None]

# ===============================================================
# Simple Linear Model
# ===============================================================
class LinearModel(nn.Module):
    num_output: int
    init_std: float = 0.1
        
    @nn.compact
    def __call__(self, x):
        return nn.Dense(features=self.num_output, use_bias=True,
                          kernel_init=nn.initializers.normal(stddev=self.init_std))(x)

    
    
# ===============================================================
# MPO
# ===============================================================

def contract_mpo(tensors):
    """
    Contract a list of MPO tensors to form the full weight matrix.
    
    Each MPO tensor in `tensors` is assumed to have shape (l, r, 2, 2), where the first and
    last tensors have l=1 or r=1 (boundary conditions). The full matrix is given by:
    
      W_{i1,...,i_n, j1,...,j_n} = Σ_{α_0,...,α_n} ∏_{k=0}^{n-1} A[k]^{i_k, j_k}_{α_k, α_{k+1}},
      
    where n = len(tensors). The resulting tensor is then reshaped to a matrix of shape
    (2^n, 2^n).
    """
    n = len(tensors)
    # Prepare index letters.
    # We use bond indices a0, a1, ..., a_n.
    letters = string.ascii_lowercase
    bond_letters = letters[: n + 1]
    phys_in_letters = letters[n + 1 : 2 * n + 1]
    phys_out_letters = letters[2 * n + 1 : 3 * n + 1]

    einsum_subscripts = []
    for k in range(n):
        # Each tensor gets a subscript like: a_k, a_{k+1}, i_k, j_k
        einsum_subscripts.append(f"{bond_letters[k]}{bond_letters[k+1]}{phys_in_letters[k]}{phys_out_letters[k]}")

    # Build the full einsum string.
    # The result will have indices i0, ..., i_{n-1}, j0, ..., j_{n-1}
    einsum_str = ",".join(einsum_subscripts) + "->" + "".join(phys_in_letters + phys_out_letters)
    # Contract the MPO.
    W = jnp.einsum(einsum_str, *tensors)
    # Reshape the resulting 2^n x 2^n operator.
    W = jnp.reshape(W, (2 ** n, 2 ** n))
    return W


def mpo_dense_layer(x, mpo_weights, num_features, bond_dim):
    """
    Apply an MPO-based linear layer to the inputs.
    
    Args:
      x: Input tensor of shape (batch_size, num_features).
      mpo_weights: 1D JAX array containing all MPO parameters.
      num_features: Dimension of the input/output vector (must be 2^n).
      bond_dim: Bond dimension of the MPO.
      
    Returns:
      The output of applying the MPO weight matrix to x.
    """
    # Determine the number of MPO tensors.
    n = int(math.log2(num_features))
    # Define the shapes for each MPO tensor.
    shapes = []
    if n == 1:
        shapes.append((1, 1, 2, 2))
    else:
        # First tensor: shape (1, bond_dim, 2, 2)
        shapes.append((1, bond_dim, 2, 2))
        # Middle tensors, if any.
        for _ in range(1, n - 1):
            shapes.append((bond_dim, bond_dim, 2, 2))
        # Last tensor: shape (bond_dim, 1, 2, 2)
        shapes.append((bond_dim, 1, 2, 2))
    
    # Split and reshape the single mpo_weights vector into the MPO tensors.
    tensors = []
    offset = 0
    for shape in shapes:
        size = np.prod(shape)
        tensor = jnp.reshape(mpo_weights[offset : offset + size], shape)
        tensors.append(tensor)
        offset += size
    
    # Contract the MPO tensors into the full weight matrix W (shape: (num_features, num_features)).
    W = contract_mpo(tensors)
    # Apply the weight matrix.
    return x @ W


class MPO_SquareLinearModel(nn.Module):
    num_features: int = 8  # Must be a power of 2.
    num_output: int = 1
    init_std: float = 0.1
    bond_dim: int = 2  # Hyperparameter for the MPO bond dimension.
    
    @nn.compact
    def __call__(self, x):
        # The input x is expected to have shape (batch, num_features).
        
        # Compute number of MPO tensors.
        n = int(math.log2(self.num_features))
        # Compute the total number of parameters needed:
        if n == 1:
            total_params = 1 * 1 * 2 * 2  # = 4
        else:
            # First MPO tensor: 1 * bond_dim * 2 * 2
            first = 4 * self.bond_dim
            # Middle MPO tensors: (n - 2) tensors each of size bond_dim * bond_dim * 2 * 2.
            middle = (n - 2) * (4 * self.bond_dim ** 2) if n > 2 else 0
            # Last MPO tensor: bond_dim * 1 * 2 * 2.
            last = 4 * self.bond_dim
            total_params = first + middle + last

        # Initialize the MPO parameters as a single vector.
        mpo_weights = self.param(
            "mpo_weights",
            lambda rng: jax.random.normal(rng, (total_params,)) * self.init_std
        )
        
        # Replace the first dense layer with an MPO-based linear layer.
        # This acts as a linear mapping from R^(num_features) to R^(num_features).
        x = mpo_dense_layer(x, mpo_weights, self.num_features, self.bond_dim)
        # Follow with the nonlinearity (elementwise square).
        x = jnp.square(x)
        # Apply the final dense layer.
        output = nn.Dense(
            features=self.num_output,
            use_bias=True,
            kernel_init=nn.initializers.normal(stddev=self.init_std)
        )(x)
        return output

    
# ===============================================================
# TTN
# ===============================================================

class TTN_FNN(nn.Module):
    num_features: int                     # number of input features (must be a power of 2)
    bond_dim: int                         # bond dimension for internal TTN tensors
    num_frequencies: int = 1              # frequencies for feature mapping
    num_output: int = 1                   # output dimension
    init_std: float = 0.1                 # stddev for TTN tensor initialization
    frequency_min_init: float = 1.0       # initial minimum frequency scaling
    trainable_frequency_min: bool = True  # whether frequency_min is a trainable parameter

    @nn.compact
    def __call__(self, x):
        # x: shape (batch, num_features)
        B = x.shape[0]
        # Ensure num_features is a power of 2
        assert (self.num_features & (self.num_features - 1)) == 0, \
            "num_features must be a power of 2 for TTN"

        # Frequency scaling parameter
        frequency_min = (
            self.param("frequency_min", 
                       lambda rng: jnp.array(self.frequency_min_init, dtype=x.dtype))
            if self.trainable_frequency_min else jnp.array(self.frequency_min_init, dtype=x.dtype)
        )

        # Physical dimension per site: 1 + 2 * num_frequencies
        D_phys = 1 + 2 * self.num_frequencies

        # Feature mapping: [1, sin(f*x), cos(f*x)]
        x_scaled = x * frequency_min
        x_exp = x_scaled[..., None]
        freqs = jnp.arange(1, self.num_frequencies + 1, dtype=x.dtype)
        sin_vals = jnp.sin(x_exp * freqs)
        cos_vals = jnp.cos(x_exp * freqs)
        ones = jnp.ones((B, self.num_features, 1), dtype=x.dtype)
        features = jnp.concatenate([ones, sin_vals, cos_vals], axis=-1)  # (B, num_features, D_phys)

        # Initialize leaf nodes
        nodes = [features[:, i, :] for i in range(self.num_features)]  # list of (B, D_phys)

        # Build the binary tree
        level = 0
        while len(nodes) > 1:
            new_nodes = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1]
                d_left = left.shape[-1]
                d_right = right.shape[-1]
                # Output dimension: bond_dim for intermediate, num_output at root
                d_out = self.num_output if len(nodes) == 2 else self.bond_dim

                # Define TTN tensor for this merge node
                A = self.param(
                    f"ttn_{level}_{i//2}",
                    nn.initializers.normal(stddev=self.init_std),
                    (d_left, d_right, d_out)
                )
                # Contract: (B, d_left) x (d_left, d_right, d_out) x (B, d_right) -> (B, d_out)
                merged = jnp.einsum("bi,ijd,bj->bd", left, A, right)
                new_nodes.append(merged)
            nodes = new_nodes
            level += 1

        # Final output at root
        return nodes[0]  # shape (B, num_output)
    
# ===============================================================
# Random Fourier Features
# =============================================================== 
class RFF(nn.Module):
    hidden_dim: int                   # hidden dimension for first linear layer
    num_output: int                   # output dimension
    init_std_hidden: float = 0.1      # stddev for first linear layer init
    init_std_output: float = 0.05     # stddev for final linear layer init

    @nn.compact
    def __call__(self, x):
        # 1. Linear layer: x1 -> x2
        x2 = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.normal(stddev=self.init_std_hidden)
        )(x)

        # 2. Trig feature mapping: x2 -> x3
        sin_feats = jnp.sin(x2)
        cos_feats = jnp.cos(x2)
        x3 = jnp.concatenate([sin_feats, cos_feats], axis=-1)

        # 3. Final linear layer: x3 -> output
        out = nn.Dense(
            features=self.num_output,
            use_bias=True,
            kernel_init=nn.initializers.normal(stddev=self.init_std_output)
        )(x3)
        return out
