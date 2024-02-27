from architectures.Transmitters.Transmitter import Transmitter


class TransmitterIdentity(Transmitter):
    def __init__(self, device):
        super().__init__(device)

    def transmit(self, x):
        """
        clear communication channel without loss
        """
        return x
