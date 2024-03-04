import torch
import helper
import presets
from toy_data import data


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

    embedding_framework = presets.preset_variance_vae(device, 3, 2, data_loader)
    embedding_framework.train(epochs=5, plot_interval=100)
    embedding_framework.plot_latent(f"images/latent.png")

    """
    TODO:
    - implement k-head vae into framework (done)
    - implement variance exploration vae into framework (done)
    - implement T-SNE
    - remove old files (done)
    - compatibility check?
    - add some extensions for visualisation??
    - support more data input??
    - make software package??
    """
