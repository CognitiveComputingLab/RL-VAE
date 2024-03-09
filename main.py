import torch
import helper
import presets
from toy_data import data
from rl_embeddings.property_calculators import PropertyCalculatorTSNE


def get_device():
    torch.manual_seed(0)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    device = get_device()

    toy_data = data.Sphere3D(n=100).generate()
    toy_dataset = helper.ToyTorchDataset(toy_data)
    data_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=5,
        shuffle=False
    )

    embedding_framework = presets.preset_tsne(device, 3, 2, data_loader)
    embedding_framework.disable_tqdm = False
    embedding_framework.train_model(epochs=5, plot_interval=20)
    embedding_framework.plot_latent(f"images/latent.png")

    """
    TODO:
    - think of optimiser / reward system (done)
    - convert reward calculators (done)
    - convert embedding_framework class (done)
    - cleanup
        - eval (done)
        - tqdm (done)
        - etc.
    - implement t-sne
    - more generality in some of the classes?
    - compatibility check?
    - module
    """
