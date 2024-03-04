import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.nn import Module
from architectures.PropertyCalculators.PropertyCalculator import PropertyCalculator
from architectures.samplers import Sampler
from architectures.Explorers.Explorer import Explorer
from architectures.transmitters import Transmitter
from architectures.RewardCalculators.RewardCalculator import RewardCalculator


class EmbeddingFramework:
    def __init__(self, device):
        # init general
        self.__device = device
        self.__disable_tqdm = False

        # components
        self.__property_calculator = None
        self.__sampler = None
        self.__encoder_agent = None
        self.__explorer = None
        self.__transmitter = None
        self.__decoder_agent = None
        self.__reward_calculator = None

        component_names = ['property_calculator', 'sampler', 'explorer', 'transmitter', 'reward_calculator',
                           'encoder_agent', 'decoder_agent']
        for name in component_names:
            self._create_property_for_component(name)

        # non-class components
        self.__property_optimizer = None
        self.__reconstruction_optimizer = None

    #######################
    # getters and setters #
    #######################

    @property
    def disable_tqdm(self):
        return self.__disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value):
        self.disable_tqdm = value

    def _create_property_for_component(self, name):
        """
        create getters and setters for all components
        """
        private_name = f"__{name}"

        def getter(instance):
            return getattr(instance, private_name)

        def setter(instance, value):
            expected_type_dict = {'property_calculator': PropertyCalculator, 'sampler': Sampler, 'explorer': Explorer,
                                  'encoder_agent': Module, 'decoder_agent': Module, 'transmitter': Transmitter,
                                  'reward_calculator': RewardCalculator}

            # check correct type
            if not isinstance(value, expected_type_dict[name]):
                raise ValueError(f"Passed non {expected_type_dict[name].__name__} object as {name}.")

            setattr(instance, private_name, value)

        setattr(self.__class__, name, property(getter, setter))

    def set_learning_mode(self, encoder_reconstruction):
        """
        define how the neural networks (encoder / decoder) should learn
        :param encoder_reconstruction: boolean, should the encoder be trained on the reconstruction reward
        """
        if not self.encoder_agent or not self.decoder_agent:
            raise ValueError("The Encoder and Decoder need to be set, before the learning mode is set.")

        # set the optimizers
        if encoder_reconstruction:
            self.__reconstruction_optimizer = torch.optim.Adam(
                list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()))
            self.__property_optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()))
        else:
            self.__reconstruction_optimizer = torch.optim.Adam(list(self.decoder_agent.parameters()))
            self.__property_optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()))

    def check_completeness(self):
        """
        check if all components are set
        check if all components are compatible
        raises an error if anything is wrong
        """
        if not self.property_calculator:
            raise ValueError("PropertyCalculator Object has not been set.")
        if not self.sampler:
            raise ValueError("Sampler Object has not been set.")
        if not self.encoder_agent:
            raise ValueError("Encoder Object has not been set.")
        if not self.explorer:
            raise ValueError("Explorer Object has not been set.")
        if not self.transmitter:
            raise ValueError("Transmitter Object has not been set.")
        if not self.decoder_agent:
            raise ValueError("Decoder Object has not been set.")
        if not self.reward_calculator:
            raise ValueError("Reward Calculator has not been set.")
        if not self.__reconstruction_optimizer or not self.__property_optimizer:
            raise ValueError("Learning mode has not been set.")

    ####################
    # training process #
    ####################

    def train(self, epochs=100, plot_interval=50):
        """
        train the encoder to produce an embedding for the given dataset
        :param epochs: number of epochs to train for as int
        :param plot_interval: each nth epoch to plot latent for
        """
        # before running, check if everything is set up correctly
        self.check_completeness()

        # distance calculation for high dimensional data
        self.property_calculator.calculate_high_dim_property()

        for epoch in tqdm(range(epochs), disable=self.disable_tqdm):
            # tell the sampler that a new epoch is starting
            self.sampler.reset_epoch()

            # run through epoch
            while not self.sampler.epoch_done:
                self.run_iteration(epoch)
                return

            # plot latent space
            if epoch % plot_interval == 0:
                self.plot_latent(f"images/latent_{epoch}.png")

    def run_iteration(self, epoch):
        """
        run single iteration of training loop
        :param epoch: current training epoch
        """
        # get batch of points
        sample_out = self.sampler(self.property_calculator.high_dim_property)
        print("sample out: ", sample_out)

        # pass through encoder
        # encoder_out = self.encoder_agent(sample_out)

        return
        z_a = self.explorer.get_point_from_output(out, epoch)

        # get complementary indices corresponding to p1
        ind2 = self.sampler.next_complementary_indices(self.property_calculator)
        property_loss = 0
        if ind2 is not None:
            # get points
            x_a2, _ = self.sampler.get_points_from_indices(ind2)
            x_a2 = x_a2.to(self.__device)

            # pass through encoder
            z_a2 = self.encoder_agent(x_a2)
            z_a2 = self.explorer.get_point_from_output(z_a2)

            # compare high and low dims
            low_dim_prop = self.property_calculator.get_low_dim_property(z_a, z_a2)
            high_dim_prop = self.property_calculator.high_dim_property[ind, ind2].to(self.__device)
            property_loss = -self.reward_calculator.calculate_property_reward(high_dim_prop, low_dim_prop)

        # communicate through transmission channel
        z_b = self.transmitter(z_a)

        # pass through decoder
        x_b = self.decoder_agent(z_b)

        # compare original point and reconstructed point
        reconstruction_loss = -self.reward_calculator.calculate_reconstruction_reward(x_a, x_b, out, self.explorer)

        # train the encoder and decoder
        total_loss = reconstruction_loss + property_loss
        self.__reconstruction_optimizer.zero_grad()
        self.__property_optimizer.zero_grad()
        total_loss.backward()
        self.__reconstruction_optimizer.step()
        self.__property_optimizer.step()

    #################
    # visualisation #
    #################

    def plot_latent(self, path):
        """
        draw every point in lower dimensional space
        only 2D is supported
        :param path: save visualisation to this directory
        """
        # init
        self.sampler.reset_epoch()
        self.explorer.evaluation_active = True

        while not self.sampler.epoch_done:
            # get batch of points
            ind = self.sampler.next_batch_indices()
            x, y = self.sampler.get_points_from_indices(ind)
            x = x.to(self.__device)

            # pass through encoder and get points
            out = self.encoder_agent(x)
            z = self.explorer.get_point_from_output(out)
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
        self.explorer.evaluation_active = False

