## DeformableODE
🌀 Transformer-based Neural ODE for Object Trajectory Modeling

This project implements a **Neural ODE model with a Transformer encoder** to learn and simulate multi-object dynamics from sequential data. The model is trained on position sequences and learns to predict full trajectories from partial observations.

---

## 🧠 Model Architecture

### 1. **Transformer Encoder**

* **Input shape**: `[N, K, T, F]`
  where:

  * `N`: number of systems
  * `K`: number of objects
  * `T`: number of time steps
  * `F`: feature dimension (e.g., 2D or 3D positions)
* Each object trajectory is embedded and processed by a Transformer encoder with positional encoding.
* Outputs a latent representation of shape `[N, K, D]`.

### 2. **Neural ODE**

* Parameterizes time evolution in latent space:

  $$
  \frac{dz}{dt} = f(z)
  $$
* Uses `torchdiffeq.odeint` to simulate continuous dynamics over time from the initial latent state `z₀`.

### 3. **Decoder**

* Maps latent trajectories back to the observed space.
* Outputs full predicted trajectories: `[N, K, T, F]`.

---

## 📊 Dataset

Please read the dataset generator and preprocess it into the required format

## 🏋️ Training & Evaluation

### Training Loop

* Uses MSE loss between predicted and ground truth trajectories.
* Only the **first 10 time steps** of each input sequence are used for context.

### Evaluation

* `evaluate`: computes average MSE across the entire test set.
* `evaluate_t`: computes and saves MSE **per time step** as `mse_results.npy`.

---

## 🔧 Configuration & Usage

### Key Parameters
Set the Parameters for your setting
```python
input_dim = F = 3
model_dim = 64
output_dim = F = 3
epochs = 100
batch_size = 4
device = 'cuda:1'  # if available
```

### Main Script Logic

1. Load data from `combined_positions.npy`
2. Split into 80% training and 20% testing
3. Initialize and train the model
4. Evaluate at intervals and save per-timestep metrics

---

## 📁 Key Components

| Component            | Description                                        |
| -------------------- | -------------------------------------------------- |
| `PositionalEncoding` | Adds sinusoidal encoding to capture time structure |
| `TransformerEncoder` | Encodes observed time-series with attention        |
| `ODEFunc`            | Parameterizes dynamics in latent space             |
| `ODEDecoder`         | Decodes latent states back to observable features  |
| `NeuralODEModel`     | Full pipeline: encode → simulate → decode          |
| `evaluate_t`         | Tracks prediction accuracy over time               |

---

## ✅ Applications

* Physical dynamics simulation (e.g., particles, pendulums, springs)
* Sequence modeling with continuous-time latent dynamics
* Learning from partially observed trajectory data

---

## Example Output

Once trained, the model predicts smooth trajectories given partial observations. The `evaluate_t` function can be used to analyze per-timestep prediction error and save results for further visualization.

## 📈 Future work
Adding graph learning based on closest particles instead of the full-connected graph (equally to MLP) could help to capture the cloest relation better. And adding physicial inductive bias would also be a good direction.
