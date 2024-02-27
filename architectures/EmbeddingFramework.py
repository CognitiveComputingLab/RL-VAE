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
                # get batch of points
                ind = self.sampler.next_batch_indices()
                x, _ = self.sampler.get_points_from_indices(ind)
                x = x.to(self.device)

                # pass through encoder
                out1 = self.encoder_agent(x)
                out1 = self.explorer.get_point_from_output(out1)

                # get complementary indices corresponding to p1
                ind2 = self.sampler.next_complementary_indices(self.property_calculator)
                if ind2:
                    # get points
                    x2, _ = self.sampler.get_points_from_indices(ind2)
                    x2 = x2.to(self.device)

                    # pass through encoder
                    out2 = self.encoder_agent(x2)
                    out2 = self.explorer.get_point_from_output(out2)

                    # calculate low dim property
                    low_dim_property = self.property_calculator.get_low_dim_property(out1, out2)
                    high_prob = self.property_calculator.high_dim_property[ind, ind2]

                return