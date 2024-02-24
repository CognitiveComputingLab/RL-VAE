

class EmbeddingFramework:
    def __init__(self, device, input_dim, output_dim):
        # init general
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        # components
        self.sampling = None

    def train(self, training_data_loader, epochs=100):
        """
        train the encoder to produce an embedding for the given dataset
        :param training_data_loader: pytorch dataloader
        :param epochs: number of epochs to train for (iterations through entire dataset)
        """
        # distance calculation for high dimensional data

        for x, _ in training_data_loader:
            x_a = x.to(self.device)





