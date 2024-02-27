class EmbeddingFramework:
    def __init__(self, device, input_dim, output_dim, data_loader):
        # init general
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        # components
        self.property_calculator = None
        self.sampler = None
        self.encoder_agent = None
        self.explorer = None
        self.transmitter = None
        self.decoder = None
        self.reward_calculator = None

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
        if not self.decoder:
            raise ValueError("Decoder Object has not been set.")
        if not self.reward_calculator:
            raise ValueError("Reward Calculator has not been set.")

    def train(self, epochs=100):
        """
        train the encoder to produce an embedding for the given dataset
        :param epochs: number of epochs to train for as int
        """
        # before running, check if everything is set up correctly
        self.check_completeness()

        # distance calculation for high dimensional data
        self.property_calculator.calculate_high_dim_property()

        for epoch in range(epochs):
            # tell the sampler that a new epoch is starting
            self.sampler.reset_epoch()

            while not self.sampler.epoch_done:
                self.run_iteration()
                return

    def run_iteration(self):
        """
        run single iteration of training loop
        """
        # get batch of points
        ind = self.sampler.next_batch_indices()
        x_a, _ = self.sampler.get_points_from_indices(ind)
        x_a = x_a.to(self.device)

        # pass through encoder
        out = self.encoder_agent(x_a)
        z_a = self.explorer.get_point_from_output(out)

        # get complementary indices corresponding to p1
        ind2 = self.sampler.next_complementary_indices(self.property_calculator)
        if ind2 is not None:
            # get points
            x_a2, _ = self.sampler.get_points_from_indices(ind2)
            x_a2 = x_a2.to(self.device)

            # pass through encoder
            z_a2 = self.encoder_agent(x_a2)
            z_a2 = self.explorer.get_point_from_output(z_a2)

            # compare high and low dims
            low_dim_prop = self.property_calculator.get_low_dim_property(z_a, z_a2)
            high_dim_prop = self.property_calculator.high_dim_property[ind, ind2]
            property_loss = -self.reward_calculator.calculate_property_reward(high_dim_prop, low_dim_prop)

        # communicate through transmission channel
        z_b = self.transmitter.transmit(z_a)

        # pass through decoder
        x_b = self.decoder(z_b)

        # compare original point and reconstructed point
        reconstruction_loss = -self.reward_calculator.calculate_reconstruction_reward(x_a, x_b, out)
