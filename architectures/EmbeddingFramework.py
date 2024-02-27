import torch
from torch.nn import Module
from architectures.PropertyCalculators.PropertyCalculator import PropertyCalculator
from architectures.Samplers.Sampler import Sampler
from architectures.Explorers.Explorer import Explorer
from architectures.Transmitters.Transmitter import Transmitter
from architectures.RewardCalculators.RewardCalculator import RewardCalculator


class EmbeddingFramework:
    def __init__(self, device):
        # init general
        self.__device = device

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

        # additional
        self.__property_optimizer = None
        self.__reconstruction_optimizer = None

    #######################
    # getters and setters #
    #######################

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
        if not self.__encoder_agent or not self.__decoder_agent:
            raise ValueError("The Encoder and Decoder need to be set, before the learning mode is set.")

        # set the optimizers
        if encoder_reconstruction:
            self.__reconstruction_optimizer = torch.optim.Adam(
                list(self.__encoder_agent.parameters()) + list(self.__decoder_agent.parameters()))
            self.__property_optimizer = torch.optim.Adam(list(self.__encoder_agent.parameters()))
        else:
            self.__reconstruction_optimizer = torch.optim.Adam(list(self.__decoder_agent.parameters()))
            self.__property_optimizer = torch.optim.Adam(list(self.__encoder_agent.parameters()))

    def check_completeness(self):
        """
        check if all components are set
        check if all components are compatible
        raises an error if anything is wrong
        """
        if not self.__property_calculator:
            raise ValueError("PropertyCalculator Object has not been set.")
        if not self.__sampler:
            raise ValueError("Sampler Object has not been set.")
        if not self.__encoder_agent:
            raise ValueError("Encoder Object has not been set.")
        if not self.__explorer:
            raise ValueError("Explorer Object has not been set.")
        if not self.__transmitter:
            raise ValueError("Transmitter Object has not been set.")
        if not self.__decoder_agent:
            raise ValueError("Decoder Object has not been set.")
        if not self.__reward_calculator:
            raise ValueError("Reward Calculator has not been set.")
        if not self.__reconstruction_optimizer or not self.__property_optimizer:
            raise ValueError("Learning mode has not been set.")

    ####################
    # training process #
    ####################

    def train(self, epochs=100):
        """
        train the encoder to produce an embedding for the given dataset
        :param epochs: number of epochs to train for as int
        """
        # before running, check if everything is set up correctly
        self.check_completeness()

        # distance calculation for high dimensional data
        self.__property_calculator.calculate_high_dim_property()

        for epoch in range(epochs):
            # tell the sampler that a new epoch is starting
            self.__sampler.reset_epoch()

            while not self.__sampler.epoch_done:
                self.run_iteration()
                return

    def run_iteration(self):
        """
        run single iteration of training loop
        """
        # reset optimizers
        self.__property_optimizer.zero_grad()
        self.__reconstruction_optimizer.zero_grad()

        # get batch of points
        ind = self.__sampler.next_batch_indices()
        x_a, _ = self.__sampler.get_points_from_indices(ind)
        x_a = x_a.to(self.__device)

        # pass through encoder
        out = self.__encoder_agent(x_a)
        z_a = self.__explorer.get_point_from_output(out)

        # get complementary indices corresponding to p1
        ind2 = self.__sampler.next_complementary_indices(self.__property_calculator)
        if ind2 is not None:
            # get points
            x_a2, _ = self.__sampler.get_points_from_indices(ind2)
            x_a2 = x_a2.to(self.__device)

            # pass through encoder
            z_a2 = self.__encoder_agent(x_a2)
            z_a2 = self.__explorer.get_point_from_output(z_a2)

            # compare high and low dims
            low_dim_prop = self.__property_calculator.get_low_dim_property(z_a, z_a2)
            high_dim_prop = self.__property_calculator.high_dim_property[ind, ind2]
            property_loss = -self.__reward_calculator.calculate_property_reward(high_dim_prop, low_dim_prop)
            property_loss.backward()

        # communicate through transmission channel
        z_b = self.__transmitter.transmit(z_a)

        # pass through decoder
        x_b = self.__decoder_agent(z_b)

        # compare original point and reconstructed point
        reconstruction_loss = -self.__reward_calculator.calculate_reconstruction_reward(x_a, x_b, out)
        reconstruction_loss.backward()

        # train the encoder and decoder
        self.__property_optimizer.step()
        self.__reconstruction_optimizer.step()

