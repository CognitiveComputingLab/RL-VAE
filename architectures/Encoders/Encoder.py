

class Encoder:
    def __init__(self, device):
        self.device = device

    """
    encoder with actual neural network
    also contains the exploration vs exploitation
        - this will additionally be a separate class    
    """