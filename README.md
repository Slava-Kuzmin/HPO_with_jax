# Hyperparameter Optimization with JAX, Flax & Optuna

This repository demonstrates how to perform **efficient hyperparameter optimization (HPO)** for JAX/Flax models using [Optuna](https://optuna.org/). It includes utilities to:

- Train multiple model replicas in parallel using `vmap` without recompiling Jax models for new hyperparameters
- Apply per-layer initialization standard deviations
- Support custom per-parameter learning rates via `lr_dict`
- Integrate with `Optuna` for automated hyperparameter search
- Perform early stopping based on smoothed validation loss
- Visualize loss/accuracy trajectories with uncertainty bands

---

## 📦 Contents

- `HPO_example.ipynb`: Example notebook showing a full HPO pipeline
- `train_multiple_models.py`: Core training logic with `vmap`, early stopping, and loss tracking
- `training_jax.py`: Optimizer and training utilities (Lion, custom learning rate masks)
- `models_jax.py`: Flax model definitions
- `pca_datasets.py`: Utilities to load and pre-process datasets with PCA
- `README.md`: You're here.

---

## 🧪 Example: Train + Optimize

In the notebook, you will:

1. Define a custom Flax model (`FNN`, `SquareLinearModel`, etc.)
2. Build a dataset with PCA (e.g. FashionMNIST)
3. Run training for multiple seeds and replicas
4. Optimize learning rates, weight decay, and initialization scale using Optuna

```python
study.optimize(objective, n_trials=50)
```

---

## 📊 Visualization

This repo includes:

- `plot_results(...)`: Plot loss/error trajectories with mean ± std shading
- `Optuna` plots (e.g. `plot_optimization_history`) using Plotly
- Histogram-style comparison plots of final losses across runs

---

## 🔧 Requirements

- `jax`, `flax`, `optax`
- `optuna[visualization]`
- `matplotlib`, `numpy`, `scikit-learn`

To install:

```bash
pip install jax flax optax optuna[visualization] matplotlib scikit-learn
```