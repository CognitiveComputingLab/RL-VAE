from architectures.Explorers.Explorer import Explorer


class ExplorerIdentity(Explorer):
    def __init__(self, device):
        super().__init__(device)

    def get_point_from_output(self, out):
        """
        get point from encoder output
        in this case there is no exploration or other transformation
        """
        return out
