# RL-Embedding Framework

This repository contains the code implemented the research outlined in the corresponding paper: "A Unified Framework for Embedding Methods Based on Reinforcement Learning".

## Code File Structure

The code file structure is split into three main sections.


The `rl_embeddings` directory holds the implementation of the various components utilized by the parametric embedding techniques presented in the paper, aimed at condensing data into more compact, lower-dimensional forms.

The `toy_data` directory is responsible for managing data operations, including importing data from files, generating data, and arranging it into the appropriate format.
Non-parametric embedding methods used for comparison are also implemented in this folder.

The `examples.py` and `main.py` files show examples of how to implement and use various embedding methods.

```
project_code/
├─ rl_embeddings/
│  ├─ components.py
│  ├─ decoders.py
│  ├─ encoders.py
│  ├─ errors.py
│  ├─ explorers.py
│  ├─ pre_trainers.py
│  ├─ samplers.py
│  ├─ reward_calculators.py
│  ├─ similarity_calculators.py
│  ├─ transmitters.py
├─ toy_data/
│  ├─ embedding.py
│  ├─ data.py
│  ├─ environments.py
│  ├─ plotting.py
│  ├─ toy_torch_dataset.py
│  ├─ util.py
├─ examples.py
├─ main.py
```

## Implementations

The `examples.py` file contains various parametric embedding methods implemented as PyTorch Module.
This includes
- VAE
- VAE-UMAP
- K-Head-VAE
  - K-Head-VAE-Decreasing (epsilon-greedy exploration)
- Variance-VAE
  - Variance-VAE-Decreasing (epsilon-greedy exploration)
- P-UMAP
- P-TSNE
- TSNE-UMAP

The `data.py` file contains various options for datasets for running the embedding methods.
This includes
- Moebius
- Sphere
- COIL20
- F-MNIST

The Moebius and Sphere datasets are artificially generated using the `ToyData.generate()` function.
The COIL20 and F-MNIST datasets will need to be downloaded before running.

## Correspondence to Paper

You will notice if you read the paper that the RL-Embedding framework components do not match up.
In the code they have been further split into subcomponents to make implementation of various methods easier.
Here is an overview of which code subcomponent corresponds to each component introduced in the paper:
- Data Distribution
  - `samplers.py`
- Encoder Policy
  - `encoders.py`
  - `explorers.py`
- Transmission Distribution
  - `transmitters.py`
- Decoder Policy
  - `decoders.py`
- Reward Function
  - `similarity_calculators.py`
  - `reward_calculators.py`

The `components.py` file contains an abstract super class for all components.
The `errors.py` file checks for compatibility errors between components.
The `pre_trainers.py` file can be used to pretrain any method on e.g Spectral embedding.

## Reproducing Results

The experiment section of the corresponding paper shows experiments on four datasets that can be accessed in the `main.py` file as:
```python
# Moebius Strip
toy_data = data.MoebiusStrip(turns=1, n=10000).generate()

# 3D Sphere
toy_data = data.Sphere3D(n=10000).generate()

# COIL-20
toy_data = data.Coil20(n=10000).generate()

# Fashion MNIST
toy_data = data.FashionMNIST(n=10000).generate()
```

There are a total of eight methods compared in the paper. Seven of them are implemented through the framework in the `examples.py` file.
They can be accessed and trained in the `main.py` file by choosing one of the following:
```python
# VAE
model = examples.VAE(input_dim, latent_dim, device, data_loader)

# V-VAE
model = examples.VarianceVAE(input_dim, latent_dim, device, data_loader)

# KH-VAE
model = examples.KHeadVAEDecreasing(input_dim, latent_dim, device, data_loader, k=5)

# VAE-UMAP
model = examples.VAE_UMAP(input_dim, latent_dim, device, data_loader)

# P-UMAP
model = examples.UMAP(input_dim, latent_dim, device, data_loader)

# TSNE-UMAP
model = examples.TSNE_UMAP(input_dim, latent_dim, device, data_loader)

# P-TSNE
model = examples.TSNE(input_dim, latent_dim, device, data_loader)
```

The UMAP comparison can simply be accessed by calling the `compare_umap` function.

The main hyperparameters need to be tuned according to the method and the dataset:

**VAE**
- **Moebius**: success-weight: 0.8, kl-weight: 0.2
- **Sphere**: success-weight: 1, kl-weight: 0.1
- **COIL20**: success-weight: 0.5, kl-weight: 0.2
- **F-MNIST**: success-weight: 0.5, kl-weight: 0.2

**V-VAE**
- **Moebius**: success-weight: 1
- **Sphere**: success-weight: 1
- **COIL20**: success-weight: 1
- **F-MNIST**: success-weight: 1

**KH-VAE**
- **Moebius**: decreasing-exploration, min-exploration: 0.01, decay-rate: 0.999, starting-exploration: 1, heads: 5
- **Sphere**: decreasing-exploration, min-exploration: 0.01, decay-rate: 0.999, starting-exploration: 1, heads: 5
- **COIL20**: decreasing-exploration, min-exploration: 0.01, decay-rate: 0.999, starting-exploration: 1, heads: 5
- **F-MNIST**: decreasing-exploration, min-exploration: 0.01, decay-rate: 0.999, starting-exploration: 1, heads: 5

**VAE-UMAP**
- **Moebius**: k-neighbours: 3, min_distance: 0.01, umap-weight: 1, kl-weight: 0.00001
- **Others**: k-neighbours: 3, min_distance: 0.01, umap-weight: 1, kl-weight: 0.0001

**P-UMAP**
- **Moebius**: k-neighbours: 15, min_distance: 0.8
- **Sphere**: k-neighbours: 15, min_distance: 0.8
- **COIL20**: k-neighbours: 3, min_distance: 0.01
- **F-MNIST**: k-neighbours: 3, min_distance: 0.01

**UMAP**
- **Moebius**: k-neighbours: 200, min_distance: 1
- **Sphere**: k-neighbours: 200, min_distance: 1
- **COIL20**: k-neighbours: 10, min_distance: 0.5
- **F-MNIST**: k-neighbours: 100, min_distance: 0.25

**TSNE-UMAP**
- **Moebius**: perplexity: 15, min_distance: 0.01, batch-size: 1000
- **Sphere**: perplexity: 5, min_distance: 0.01, batch-size: 1000
- **COIL20**: perplexity: 15, min_distance: 1, batch-size: 1000
- **F-MNIST**: perplexity: 15, min_distance: 1, batch-size: 1000

**P-TSNE**
- **Moebius**: perplexity: 30, batch-size: 1000
- **Sphere**: perplexity: 80, batch-size: 1000
- **COIL20**: perplexity: 100, batch-size: 1000
- **F-MNIST**: perplexity: 30, batch-size: 1000

Run the model for 1000 episodes to guarantee convergence and make sure to output the visualisations often to get the best episode.
This can be controlled via tha `latent_freq` parameter in the following:
```python
m = Main(model, toy_data)
m.train(epochs=1000, latent_freq=2)
m.plot_latent(f"images/post-trained.png")
```
