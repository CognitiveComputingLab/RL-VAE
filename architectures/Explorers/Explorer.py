

class Explorer:
    def __init__(self, device):
        self.device = device

    def get_point_from_output(self, out):
        """
        get single point from neural network output
        :param out: output directly from neural network component (encoder)
        """
        return out
