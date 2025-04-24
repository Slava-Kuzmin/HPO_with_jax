# train_multiple_models.py

from training_jax import *
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Tuple, Dict
from flax import struct
from functools import partial

# --- Custom Lion State & Initialization ---
@struct.dataclass
class LionState:
    m: any  # momentum accumulator (a PyTree)

def init_lion_state(params):
    return jax.tree_map(lambda p: jnp.zeros_like(p), params)

# --- Jittable Functions Defined at Module Level ---
@partial(jax.jit, static_argnums=(7))
def train_step_batch_lion(params, lion_state, batch, global_lr, wd, b1, b2, model, spec_mask_tree, spec_value_tree):
    """
    Performs a training step on a batch using a custom Lion update with per-weight learning rates.
    
    Dynamic arguments: params, lion_state, batch, global_lr, wd.
    Static arguments: b1, b2, model, spec_mask_tree, spec_value_tree.
    
    For each parameter, effective_lr = spec_value + (1 - spec_mask) * global_lr.
    Then, new_param = param - effective_lr * update, where update = sign(new_m + b2 * grad) + wd * param,
    and new_m = b1 * m + (1 - b1) * grad.
    """
    
    def single_loss_fn(p):
        smoothing = 0
        logits = model.apply({'params': p}, batch['input'])
        loss = cross_entropy_loss(logits, batch['label'], smoothing)
        return loss, logits

    grad_fn = jax.value_and_grad(single_loss_fn, has_aux=True)
    (batch_losses, logits), grads = jax.vmap(grad_fn)(params)
    

    # Unfreeze both lion_state and grads so that they are regular dicts.
    unfrozen_lion = unfreeze(lion_state)
    unfrozen_grads = unfreeze(grads)
    new_lion_dict = jax.tree_map(lambda m, g: b1 * m + (1 - b1) * g, unfrozen_lion, unfrozen_grads)
    new_lion_state = freeze(new_lion_dict)  # Convert back to FrozenDict.
    
    updates = jax.tree_map(lambda m, g, p: jnp.sign(m + b2 * g) + wd * p, new_lion_state, grads, params)
    
    # Compute effective LR per weight: effective_lr = spec_value + (1 - spec_mask) * global_lr.
    new_params = jax.tree_map(
        lambda p, u, mask, spec_val: p - (spec_val + (1.0 - mask) * global_lr) * u,
        params, updates, spec_mask_tree, spec_value_tree
    )
    return new_params, new_lion_state, batch_losses, logits




@partial(jax.jit, static_argnums=(2))
def eval_step_batch(params, batch, model):
    def single_eval_fn(p):
        logits = model.apply({'params': p}, batch['input'])
        loss = cross_entropy_loss(logits, batch['label'], smoothing = 0)
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == batch['label'])
        return loss, acc
    losses, accs = jax.vmap(single_eval_fn)(params)
    return losses, accs


def scale_init_params(params: Dict, std_dict: Dict[str, float]) -> Dict:
    """
    Given a params PyTree initialized at unit std,
    multiply each weight by std_dict[layer_name] or std_dict["default"].
    Leaves 'frequency_min' alone.
    """
    flat = flatten_dict(unfreeze(params))
    new_flat = {}
    default = std_dict["default"]

    for path, w in flat.items():
        # path is a tuple like ("Dense_0","kernel") or ("frequency_min",)
        key_full = "/".join(path)    # e.g. "Dense_0/kernel" or "frequency_min"
        layer    = path[0]           # always safe; e.g. "Dense_0" or "frequency_min"

        # priority: exact match → layer-level match → default
        if key_full in std_dict:
            scale = std_dict[key_full]
        elif layer in std_dict:
            scale = std_dict[layer]
        else:
            scale = default

        new_flat[path] = w * scale

    return freeze(unflatten_dict(new_flat))



def create_lr_mask_trees(params: Dict, lr_dict: Dict[str, float]):
    """
    As before: returns two FrozenDicts matching 'params':
      - mask_tree: 1.0 where key in lr_dict else 0.0
      - value_tree: lr_dict[key] where present else 0.0
    """
    flat = flatten_dict(unfreeze(params))
    flat_mask = {}
    flat_val  = {}
    for key, _ in flat.items():
        key_str = "/".join(key)
        if key_str in lr_dict:
            flat_mask[key] = 1.0
            flat_val[key]  = lr_dict[key_str]
        else:
            flat_mask[key] = 0.0
            flat_val[key]  = 0.0
    return (
        freeze(unflatten_dict(flat_mask)),
        freeze(unflatten_dict(flat_val))
    )


def train_multiple_cls_models_vmap_lion_dynamic(
    N: int,
    model:   nn.Module,    # your Flax nn.Module
    X_train: jnp.ndarray,
    y_train: jnp.ndarray,
    X_val:   jnp.ndarray,
    y_val:   jnp.ndarray,
    X_test:  jnp.ndarray,
    y_test:  jnp.ndarray,
    batch_size:     int,
    initial_wd:     float,
    init_std_dict:  Dict[str, float],
    lr_dict:        Dict[str, float],
    target_epochs:  int,
    patience              = np.inf,
    b1:             float  = 0.9,
    b2:             float  = 0.99,
    ema_decay:      float  = 1,
    seed:           int    = 0,
    print_output:   bool   = True,
) -> Tuple[ClassificationTrainState, list, dict]:
    """
    Train N models in parallel with Lion + early stopping,
    using per-layer init‐stds from init_std_dict and per-weight
    learning rates from lr_dict.  The lr_dict["default"] is
    used everywhere we previously passed initial_lr.
    """
    
    # require a default learning rate
    if lr_dict is None or "default" not in lr_dict:
        raise ValueError("`lr_dict` must be provided and contain a 'default' key")
    default_lr = lr_dict["default"]

    # 1) PRNGs & dummy batch for init
    rng  = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, N)
    x0   = jnp.array(X_train[:1])

    # 3) Initialize each replica, scale its params, set up Lion state
    states = []
    for i in range(N):
        p_i = model.init(rngs[i], x0)["params"]
        p_i = scale_init_params(p_i, init_std_dict)
        l_i = init_lion_state(p_i)
        s_i = create_train_state_cls(
            model, rngs[i],
            default_lr, initial_wd,
            x0, clip_by_global_norm=None
        )
        s_i = s_i.replace(params=p_i, opt_state=l_i)
        states.append(s_i)

    # freeze so tree‐structure matches for masking
    states = [s.replace(params=freeze(s.params)) for s in states]

    # 4) Build your per-weight LR masks
    spec_mask_tree, spec_value_tree = create_lr_mask_trees(states[0].params, lr_dict)

    # 5) Stack for vmap
    params_batched      = jax.tree_map(lambda *xs: jnp.stack(xs), *[s.params     for s in states])
    lion_states_batched = jax.tree_map(lambda *xs: jnp.stack(xs), *[s.opt_state for s in states])

    # Prepare history + per‐replica stopping
    metrics_history = [{
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "smoothed_val_loss": []
    } for _ in range(N)]
    
    best_val_loss = np.full((N,), np.inf)
    best_params_list = [None] * N
    best_lion_list = [None] * N
    epochs_no_improve = np.zeros(N, dtype=int)
    smoothed_val_loss = None
    
    for epoch in range(target_epochs):
        # --- Training Phase ---
        epoch_train_loss = jnp.zeros((N,))
        epoch_train_acc = jnp.zeros((N,))
        train_batches = 0
        for batch in data_iterator(X_train, y_train, batch_size, shuffle=True, seed=epoch):
            params_batched, lion_states_batched, batch_losses, logits = train_step_batch_lion(
                params_batched, lion_states_batched, batch,
                default_lr, initial_wd,
                b1, b2,
                model, spec_mask_tree, spec_value_tree
            )
            preds = jnp.argmax(logits, axis=-1)  # shape: (N, batch_size)
            batch_acc = jnp.mean(preds == batch['label'], axis=-1)  # shape: (N,)
            epoch_train_loss += batch_losses
            epoch_train_acc += batch_acc
            train_batches += 1
        epoch_train_loss /= train_batches
        epoch_train_acc /= train_batches

        # --- Validation Phase ---
        epoch_val_loss = jnp.zeros((N,))
        epoch_val_acc = jnp.zeros((N,))
        val_batches = 0
        for batch in data_iterator(X_val, y_val, batch_size, shuffle=False):
            batch_losses, batch_acc = eval_step_batch(params_batched, batch, model)
            epoch_val_loss += batch_losses
            epoch_val_acc += batch_acc
            val_batches += 1
        epoch_val_loss /= val_batches
        epoch_val_acc /= val_batches

        epoch_val_loss_np = np.array(epoch_val_loss)
        if smoothed_val_loss is None:
            smoothed_val_loss = epoch_val_loss_np.copy()
        else:
            smoothed_val_loss = ema_decay * epoch_val_loss_np + (1 - ema_decay) * smoothed_val_loss

        for i in range(N):
            metrics_history[i]["train_loss"].append(float(epoch_train_loss[i].item()))
            metrics_history[i]["train_accuracy"].append(float(epoch_train_acc[i].item()))
            metrics_history[i]["val_loss"].append(float(epoch_val_loss[i].item()))
            metrics_history[i]["val_accuracy"].append(float(epoch_val_acc[i].item()))
            metrics_history[i]["smoothed_val_loss"].append(float(smoothed_val_loss[i]))
        
        if print_output:
            msg = f"Epoch {epoch+1}/{target_epochs}: "
            for i in range(N):
                msg += f"Model {i}: Val Loss = {epoch_val_loss_np[i]:.4f} (smoothed = {smoothed_val_loss[i]:.4f}) \n "
            print(msg)
            print("")
        
        for i in range(N):
            if smoothed_val_loss[i] < best_val_loss[i]:
                best_val_loss[i] = smoothed_val_loss[i]
                best_params_list[i] = jax.tree_map(lambda p: np.array(p[i]), params_batched)
                best_lion_list[i] = jax.tree_map(lambda p: np.array(p[i]), lion_states_batched)
                epochs_no_improve[i] = 0
            else:
                epochs_no_improve[i] += 1
        if np.all(epochs_no_improve >= patience):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # --- Test Evaluation ---
    best_params_batched = jax.tree_map(lambda *xs: jnp.stack(xs), *best_params_list)
    test_loss_total = jnp.zeros((N,))
    test_acc_total = jnp.zeros((N,))
    test_batches = 0
    for batch in data_iterator(X_test, y_test, batch_size, shuffle=False):
        batch_losses, batch_acc = eval_step_batch(best_params_batched, batch, model)
        test_loss_total += batch_losses
        test_acc_total += batch_acc
        test_batches += 1
    test_loss = (test_loss_total / test_batches).tolist()
    test_accuracy = (test_acc_total / test_batches).tolist()
    test_metrics = {"test_loss": test_loss, "test_accuracy": test_accuracy}
    
    final_state = states[0].replace(params=params_batched, opt_state=lion_states_batched)
    return final_state, metrics_history, test_metrics