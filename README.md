# RL-Embedding Framework

This repository contains the code for my bachelor dissertation on creating an RL-Embedding Framework.

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