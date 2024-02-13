import torch

from toy_data import data
from toy_data import plotting
from toy_data import embedding

from architectures import rl_vae, vae, ce_rl_vae, de_rl_vae, distance_rl_vae, umap_vae
import helper


def get_device():
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


class Main:
    def __init__(self, batch_size=32, data_n=100):
        # device
        self.device = get_device()

        # hyper parameters
        self.batch_size = batch_size
        self.data_n = data_n
        self.save_model = False
        self.save_plots = True

        # data
        self.toy_data = None
        self.toy_dataset = None
        self.data_loader = None
        self.input_dim = None
        self.get_data()

    def get_data(self):
        """
        get relevant data and format into correct object classes
        """
        self.toy_data = data.MoebiusStrip(n=self.data_n, width=1, turns=1).generate()
        self.toy_dataset = helper.ToyTorchDataset(self.toy_data)
        self.data_loader = torch.utils.data.DataLoader(
            self.toy_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        self.input_dim = self.toy_data.data.shape[1]

    def plot_data(self, dimensions=2):
        """
        show data in 3d or 2d
        """
        if dimensions == 2:
            plotting.mpl_2d_plot(self.toy_data)
            self.toy_data.plot()
        elif dimensions == 3:
            fig = plotting.scatter3d(self.toy_data)
            fig.show()

    def show_umap(self):
        """
        show the real umap projection of the data in 2 dims
        """
        umap = embedding.UMAP(self.toy_data)
        umap.fit()
        umap.plot()

    def show_nn_umap(self):
        """
        show nn umap projection of data in 2 dims
        """
        umap_vae_obj = umap_vae.UMAP_VAE(self.device, self.input_dim)
        umap_vae_obj.init_umap(self.toy_data)
        umap_vae_obj.train(self.toy_data)

    def run_rl_vae(self):
        """
        run main model
        """
        success_weights = [1, 50]
        for i in success_weights:
            model = vae.VaeSystem(self.device, self.input_dim)
            # model = rl_vae.RlVae(device, input_dim, 1)
            # model = de_rl_vae.DecreasingExplorationRLVAE(device, input_dim, 3)
            # model = ce_rl_vae.ConstantExplorationRLVAE(device, input_dim)
            # model = distance_rl_vae.DistanceRLVAE(device, input_dim, 5)

            model.success_weight = i
            try:
                model.train(self.data_loader, epochs=1000)
            except KeyboardInterrupt:
                print("stopping early...")

            # save visualisations
            if self.save_plots:
                model.plot_latent(self.data_loader, f"images/{model.arch_name}-latent.png")
                model.plot_loss(f"images/{model.arch_name}-loss.png")

            # save model
            if self.save_model:
                model.save_model(f"models/")


if __name__ == "__main__":
    main_obj = Main()
    main_obj.show_nn_umap()
