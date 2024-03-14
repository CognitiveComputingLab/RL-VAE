import matplotlib.pyplot as plt
import torch
import helper
import examples
from toy_data import data
from tqdm import tqdm


def get_device():
    torch.manual_seed(0)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def train(emb_model, epochs, reward_name="total_reward"):
    optimizer = torch.optim.Adam(list(emb_model.parameters()))
    for epoch in tqdm(range(epochs), disable=False):
        emb_model.sampler.reset_epoch()

        # run through epoch
        epoch_done = False
        while not epoch_done:
            reward, epoch_done = emb_model(epoch)
            loss = -reward[reward_name]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            plot_latent(emb_model, f"images/latent-{epoch}.png")


def plot_latent(emb_model, path):
    # init
    emb_model.eval()
    emb_model.sampler.reset_epoch()

    while not emb_model.sampler.epoch_done:
        # get batch of points
        if hasattr(emb_model, 'property'):
            sample_out = emb_model.sampler(**{"high_dim_property": emb_model.property.high_dim_property})
        else:
            sample_out = emb_model.sampler()
        _, y = sample_out["points"]

        # pass through encoder and get points
        out = emb_model.encoder(**sample_out)
        if hasattr(emb_model, 'explorer'):
            out["epoch"] = 0
            out = emb_model.explorer(**out)
        z = out["encoded_points"]
        z = z.detach().to('cpu').numpy()

        # plot batch of points
        colors = y[:, :3].to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=colors)

    # generate image
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'latent projection')
    plt.savefig(path)
    plt.close()

    # un-init
    emb_model.train(True)


if __name__ == "__main__":
    device = get_device()

    toy_data = data.Sphere3D(n=1000).generate()
    toy_dataset = helper.ToyTorchDataset(toy_data)
    data_loader = torch.utils.data.DataLoader(
        toy_dataset,
        batch_size=100,
        shuffle=False
    )

    # model = examples.UMAP(3, 2, device, data_loader)
    # model = examples.VAE(3, 2, device, data_loader)
    # model = examples.TSNE(3, 2, device, data_loader)
    # model = examples.KHeadVAE(3, 2, device, data_loader, k=5)
    model = examples.KHeadVAEDecreasing(3, 2, device, data_loader, k=5)
    # model = examples.VarianceVAE(3, 2, device, data_loader)

    train(model, epochs=51, reward_name="total_reward")

    """
    TODO
    - fix TSNE
    """


