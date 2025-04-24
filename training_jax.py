# training_jax.py
"""
Unified training functions for JAX models using Flax.
Supports both classification and regression tasks.

Included:
 • Metrics and TrainState definitions for both tasks.
 • A simple MLP model.
 • Data utilities: create_data() for classification (moons dataset) and create_regression_data() for regression.
 • Unified data iterator.
 • Serialization utilities.
 • Train state creation functions for classification (create_train_state_cls) and regression (create_train_state_reg).
 • Loss functions and JIT–compiled training and metric computation steps for both tasks.
 • End–to–end training loops: train_cls() and train_reg().
 • Additional utility functions: count_parameters() and plot_metrics().
"""

import pickle
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import linen as nn
from flax import struct
from clu import metrics
from flax.training import train_state
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from functools import partial

# ================================================================
# Metrics and TrainState Definitions
# ================================================================

# Classification metrics and state
@struct.dataclass
class ClassificationMetrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

class ClassificationTrainState(train_state.TrainState):
    metrics: ClassificationMetrics

# Regression metrics and state
@struct.dataclass
class RegressionMetrics(metrics.Collection):
    loss: metrics.Average.from_output('loss')
    mse: metrics.Average.from_output('mse')

class RegressionTrainState(train_state.TrainState):
    metrics: RegressionMetrics

# ================================================================
# Model Definition (Simple MLP)
# ================================================================

class SimpleModel(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x

# ================================================================
# Data Utilities
# ================================================================

def create_data(n_samples=1000, noise=0.2, random_state=42):
    """
    Create a moons dataset for classification.
    Returns X_train, X_test, y_train, y_test, scaler.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def create_regression_data(n_samples=1000, noise=0.1, random_state=42):
    """
    Create a synthetic regression dataset.
    Returns X_train, X_test, y_train, y_test.
    """
    np.random.seed(random_state)
    X = np.random.rand(n_samples, 1) * 10.0  # features in range [0, 10]
    true_weights = np.array([2.5])
    y = X.dot(true_weights) + 3.0 + np.random.randn(n_samples, 1) * noise
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def data_iterator(features, labels, batch_size, shuffle=True, seed=0):
    """
    Yields batches as a dictionary with keys 'input' and 'label'.
    """
    size = len(features)
    indices = np.arange(size)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    for start_idx in range(0, size, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield {
            "input": jnp.array(features[batch_indices]),
            "label": jnp.array(labels[batch_indices])
        }

# ================================================================
# Serialization Utilities
# ================================================================

def save_state_to_bytes(params, opt_state):
    data = {"params": params, "opt_state": opt_state}
    return pickle.dumps(data)

def load_state_from_bytes(byte_data):
    data = pickle.loads(byte_data)
    return data["params"], data["opt_state"]

# ================================================================
# Train State Creation Functions
# ================================================================

def create_train_state_cls(module, rng, learning_rate, weight_decay, x_item, clip_by_global_norm=0.01):
    """
    Create an initial TrainState for classification tasks.
    """
    params = module.init(rng, x_item)['params']
    tx = optax.lion(learning_rate, weight_decay=weight_decay)
    if clip_by_global_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(clip_by_global_norm), tx)
    return ClassificationTrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=ClassificationMetrics.empty()
    )

def create_train_state_reg(module, rng, learning_rate, weight_decay, x_item, clip_by_global_norm=0.01):
    """
    Create an initial TrainState for regression tasks.
    """
    params = module.init(rng, x_item)['params']
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    if clip_by_global_norm is not None:
        tx = optax.chain(optax.clip_by_global_norm(clip_by_global_norm), tx)
    return RegressionTrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        metrics=RegressionMetrics.empty()
    )

# ================================================================
# Loss and Training Step Functions
# ================================================================

# --- Classification ---
def cross_entropy_loss(logits, labels, smoothing=0.0):
    num_classes = logits.shape[-1]
    if smoothing > 0:
        labels_one_hot = jax.nn.one_hot(labels, num_classes)
        smoothed_labels = (1 - smoothing) * labels_one_hot + (smoothing / num_classes)
        loss = optax.softmax_cross_entropy(logits, smoothed_labels)
    else:
        one_hot = jax.nn.one_hot(labels, num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot)
    return loss.mean()

@partial(jax.jit, static_argnums=(2,))
def train_step_cls(state, batch, smoothing=0.0):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['input'])
        loss = cross_entropy_loss(logits, batch['label'], smoothing)
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    # Use logits and labels for updating metrics.
    metric_updates = state.metrics.single_from_model_output(loss=loss, logits=logits, labels=batch['label'])
    state = state.replace(metrics=state.metrics.merge(metric_updates))
    return state, loss, None  # 'acc' is not needed here

@partial(jax.jit, static_argnums=(2,))
def compute_metrics_cls(state, batch, smoothing=0.0):
    logits = state.apply_fn({'params': state.params}, batch['input'])
    loss = cross_entropy_loss(logits, batch['label'], smoothing)
    # Use logits and labels for the Accuracy metric.
    metric_updates = state.metrics.single_from_model_output(loss=loss, logits=logits, labels=batch['label'])
    metrics_out = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics_out)
    return state, loss, None

def train_cls(state, X_train, y_train, X_val, y_val, batch_size, target_epochs,
              start_epoch=1, smoothing=0.0, print_output=False, prune_callback=None):
    """
    Training loop for classification tasks using a validation set.
    Evaluates on (X_val, y_val) during each epoch and records baseline metrics before training starts.
    """
    metrics_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    # Baseline evaluation (epoch 0)
    if start_epoch == 0:
        for batch in data_iterator(X_train, y_train, batch_size, shuffle=False):
            state, _, _ = compute_metrics_cls(state, batch, smoothing)
        baseline_train = state.metrics.compute()
        state = state.replace(metrics=state.metrics.empty())

        for batch in data_iterator(X_val, y_val, batch_size, shuffle=False):
            state, _, _ = compute_metrics_cls(state, batch, smoothing)
        baseline_val = state.metrics.compute()
        state = state.replace(metrics=state.metrics.empty())

        metrics_history["train_loss"].append(float(baseline_train["loss"]))
        metrics_history["train_accuracy"].append(float(baseline_train["accuracy"]))
        metrics_history["val_loss"].append(float(baseline_val["loss"]))
        metrics_history["val_accuracy"].append(float(baseline_val["accuracy"]))
    
    # Epoch loop: starting at start_epoch.
    for epoch in range(start_epoch, target_epochs):
        # Training phase.
        for batch in data_iterator(X_train, y_train, batch_size, shuffle=True, seed=epoch):
            state, loss, _ = train_step_cls(state, batch, smoothing)
            state, _, _ = compute_metrics_cls(state, batch, smoothing)
        train_metrics = state.metrics.compute()
        metrics_history["train_loss"].append(float(train_metrics["loss"]))
        metrics_history["train_accuracy"].append(float(train_metrics["accuracy"]))
        state = state.replace(metrics=state.metrics.empty())
        
        # Evaluation on validation set.
        for batch in data_iterator(X_val, y_val, batch_size, shuffle=False):
            state, _, _ = compute_metrics_cls(state, batch, smoothing)
        val_metrics = state.metrics.compute()
        metrics_history["val_loss"].append(float(val_metrics["loss"]))
        metrics_history["val_accuracy"].append(float(val_metrics["accuracy"]))
        state = state.replace(metrics=state.metrics.empty())
        
        if print_output:
            print(
                f"Epoch {epoch+1}/{target_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )
        if prune_callback:
            prune_callback(epoch, val_metrics["loss"])
    
    return state, metrics_history


# --- Regression ---
def mse_loss(predictions, targets):
    return jnp.mean((predictions - targets) ** 2)

@jax.jit
def train_step_reg(state, batch):
    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, batch['input'])
        loss = mse_loss(predictions, batch['label'])
        return loss, predictions
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    mse = jnp.mean((predictions - batch['label']) ** 2)
    metric_updates = state.metrics.single_from_model_output(loss=loss, mse=mse)
    state = state.replace(metrics=state.metrics.merge(metric_updates))
    return state

@jax.jit
def compute_metrics_reg(state, batch):
    predictions = state.apply_fn({'params': state.params}, batch['input'])
    loss = mse_loss(predictions, batch['label'])
    mse = jnp.mean((predictions - batch['label']) ** 2)
    metric_updates = state.metrics.single_from_model_output(loss=loss, mse=mse)
    metrics_out = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics_out)
    return state, loss, mse

def train_reg(state, X_train, y_train, X_val, y_val, batch_size, target_epochs,
              start_epoch=1, print_output=False):
    """
    Training loop for regression tasks.
    
    This version evaluates and records metrics (loss and mse) on both the training and
    validation sets before training begins (baseline at step 0).
    """
    metrics_history = {
        'train_loss': [],
        'train_mse': [],
        'val_loss': [],
        'val_mse': []
    }
    

    # Baseline evaluation (epoch 0)
    if start_epoch == 0:
       # Evaluate on training data.
        for batch in data_iterator(X_train, y_train, batch_size, shuffle=False):
            state, _, _ = compute_metrics_reg(state, batch)
        baseline_train = state.metrics.compute()
        state = state.replace(metrics=state.metrics.empty())

        # Evaluate on validation data.
        for batch in data_iterator(X_val, y_val, batch_size, shuffle=False):
            state, _, _ = compute_metrics_reg(state, batch)
        baseline_val = state.metrics.compute()
        state = state.replace(metrics=state.metrics.empty())

        metrics_history['train_loss'].append(float(baseline_train['loss']))
        metrics_history['train_mse'].append(float(baseline_train['mse']))
        metrics_history['val_loss'].append(float(baseline_val['loss']))
        metrics_history['val_mse'].append(float(baseline_val['mse']))
        
        
    # Epoch loop: starting at start_epoch.
    for epoch in range(start_epoch, target_epochs):
        # Training phase.
        for batch in data_iterator(X_train, y_train, batch_size, shuffle=True, seed=epoch):
            state = train_step_reg(state, batch)
            state, _, _ = compute_metrics_reg(state, batch)
        train_metrics = state.metrics.compute()
        metrics_history['train_loss'].append(float(train_metrics['loss']))
        metrics_history['train_mse'].append(float(train_metrics['mse']))
        state = state.replace(metrics=state.metrics.empty())
        
        # Evaluation on validation set.
        for batch in data_iterator(X_val, y_val, batch_size, shuffle=False):
            state, _, _ = compute_metrics_reg(state, batch)
        val_metrics = state.metrics.compute()
        metrics_history['val_loss'].append(float(val_metrics['loss']))
        metrics_history['val_mse'].append(float(val_metrics['mse']))
        state = state.replace(metrics=state.metrics.empty())
        
        if print_output:
            print(
                f"Epoch {epoch+1}/{target_epochs} - "
                f"Train Loss: {metrics_history['train_loss'][-1]:.5f}, "
                f"Train MSE: {metrics_history['train_mse'][-1]:.5f}, "
                f"Val Loss: {metrics_history['val_loss'][-1]:.5f}, "
                f"Val MSE: {metrics_history['val_mse'][-1]:.5f}"
            )
    
    return state, metrics_history


# ================================================================
# Utility Functions
# ================================================================

def count_parameters(params):
    """Count total number of parameters in a model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def plot_metrics(metrics_list, labels, task='regression', y_scale='log', x_scale='log'):
    """
    Plot metrics for multiple experiments.
    
    For regression: expects keys 'train_loss', 'test_loss'.
    For classification: expects keys 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy'.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if task == 'regression':
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for metrics, label in zip(metrics_list, labels):
            epochs = np.arange(1, len(metrics["train_loss"]) + 1)
            if y_scale == 'log':
                axs[0].semilogy(epochs, metrics["train_loss"], label=label)
                axs[1].semilogy(epochs, metrics["val_loss"], label=label)
            else:
                axs[0].plot(epochs, metrics["train_loss"], label=label)
                axs[1].plot(epochs, metrics["val_loss"], label=label)
        axs[0].set_title("Train Loss")
        axs[1].set_title("Validation Loss")

    elif task == 'classification':
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        for metrics, label in zip(metrics_list, labels):
            epochs = np.arange(1, len(metrics["train_accuracy"]) + 1)
            if y_scale == 'log':
                axs[0, 0].semilogy(epochs, 1 - np.array(metrics["train_accuracy"]), label=label)
                axs[0, 1].semilogy(epochs, 1 - np.array(metrics["val_accuracy"]), label=label)
                axs[1, 0].semilogy(epochs, metrics["train_loss"], label=label)
                axs[1, 1].semilogy(epochs, metrics["val_loss"], label=label)
            else:
                axs[0, 0].plot(epochs, 1 - np.array(metrics["train_accuracy"]), label=label)
                axs[0, 1].plot(epochs, 1 - np.array(metrics["val_accuracy"]), label=label)
                axs[1, 0].plot(epochs, metrics["train_loss"], label=label)
                axs[1, 1].plot(epochs, metrics["val_loss"], label=label)
        axs[0, 0].set_title("Train Error (1 - Accuracy)")
        axs[0, 1].set_title("Validation Error (1 - Accuracy)")
        axs[1, 0].set_title("Train Loss")
        axs[1, 1].set_title("Validation Loss")
        for ax in axs.flat:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.set_xscale(x_scale)
            ax.legend()
            ax.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
        return

