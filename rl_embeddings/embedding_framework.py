import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn.modules.module import T
from tqdm import tqdm
from rl_embeddings.property_calculators import PropertyCalculator
from rl_embeddings.samplers import Sampler
from rl_embeddings.explorers import Explorer
from rl_embeddings.transmitters import Transmitter
from rl_embeddings.reward_calculators import RewardCalculator


class EmbeddingFramework(nn.Module):
    def __init__(self, device):
        super().__init__()

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
        self.__encoder_optimizer = None
        self.__decoder_optimizer = None
        self.__total_optimizer = None

    #######################
    # getters and setters #
    #######################

    @property
    def disable_tqdm(self):
        return self.__disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value):
        self.__disable_tqdm = value

        if self.property_calculator:
            self.property_calculator.disable_tqdm = value

    def _create_property_for_component(self, name):
        """
        create getters and setters for all components
        """
        private_name = f"__{name}"

        def getter(instance):
            return getattr(instance, private_name)

        def setter(instance, value):
            expected_type_dict = {'property_calculator': PropertyCalculator, 'sampler': Sampler, 'explorer': Explorer,
                                  'encoder_agent': nn.Module, 'decoder_agent': nn.Module, 'transmitter': Transmitter,
                                  'reward_calculator': RewardCalculator}

            # check correct type
            if not isinstance(value, expected_type_dict[name]):
                raise ValueError(f"Passed non {expected_type_dict[name].__name__} object as {name}.")

            # handle tqdm
            if name == 'property_calculator':
                value.disable_tqdm = self.disable_tqdm

            setattr(instance, private_name, value)

        setattr(self.__class__, name, property(getter, setter))

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

    def init_optimizers(self):
        """
        initialize the optimizers for encoder and decoder training
        """
        # init optimizers
        self.__encoder_optimizer = torch.optim.Adam(list(self.encoder_agent.parameters()))
        self.__decoder_optimizer = torch.optim.Adam(list(self.decoder_agent.parameters()))
        self.__total_optimizer = torch.optim.Adam(
            list(self.encoder_agent.parameters()) + list(self.decoder_agent.parameters()))

    ####################
    # training process #
    ####################

    def train_model(self, epochs=100, plot_interval=50):
        """
        train the encoder to produce an embedding for the given dataset
        :param epochs: number of epochs to train for as int
        :param plot_interval: each nth epoch to plot latent for
        """
        # before running, check if everything is set up correctly
        self.check_completeness()
        self.init_optimizers()

        # distance calculation for high dimensional data
        self.property_calculator.calculate_high_dim_property()

        print(f"Starting training for {epochs} epochs")
        for epoch in tqdm(range(epochs), disable=self.__disable_tqdm):
            # tell the sampler that a new epoch is starting
            self.sampler.reset_epoch()

            # run through epoch
            while not self.sampler.epoch_done:
                self.forward(epoch)

            # plot latent space
            if epoch % plot_interval == 0:
                self.plot_latent(f"images/latent_{epoch}.png")

    def forward(self, epoch):
        """
        run single iteration of training loop
        :param epoch: current training epoch
        """
        # get batch of points
        sample_out = self.sampler(self.property_calculator.high_dim_property)

        # pass through encoder network
        encoder_out = self.encoder_agent(sample_out)

        # choose action based on exploration
        explorer_out = self.explorer(encoder_out, epoch)

        # compute low and high dimensional properties
        property_out = self.property_calculator(explorer_out)

        # communicate through transmission channel
        transmitter_out = self.transmitter(explorer_out)

        # detach output from computational graph
        if type(transmitter_out) is tuple:
            transmitter_out_detached = tuple(t.detach() for t in transmitter_out)
        else:
            transmitter_out_detached = transmitter_out.detach()

        # pass through decoder network
        decoder_out = self.decoder_agent(transmitter_out_detached)

        # get rewards / loss for training
        encoder_reward, decoder_reward, t_reward = self.reward_calculator(sample_out, encoder_out, explorer_out,
                                                                          property_out, transmitter_out, decoder_out)
        encoder_loss = -encoder_reward
        decoder_loss = -decoder_reward
        total_loss = -t_reward

        # train the encoder on encoder_loss
        self.__encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.__encoder_optimizer.step()

        # train decoder on decoder_loss
        self.__decoder_optimizer.zero_grad()
        decoder_loss.backward()
        self.__decoder_optimizer.step()

        # jointly train encoder and decoder on total_loss
        self.__total_optimizer.zero_grad()
        total_loss.backward()
        self.__total_optimizer.step()

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
        self.eval()

        while not self.sampler.epoch_done:
            # get batch of points
            sample_out = self.sampler(self.property_calculator.high_dim_property)
            p1 = sample_out
            if len(p1) > 2:
                p1 = p1[0]
            _, y = p1

            # pass through encoder and get points
            encoder_out = self.encoder_agent(sample_out)
            explorer_out = self.explorer(encoder_out, epoch=0)
            z = explorer_out
            if type(z) is tuple:
                z = z[0]
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
        self.train(True)
