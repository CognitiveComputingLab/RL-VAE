from architectures.Explorers.Explorer import Explorer


class ExplorerIdentity(Explorer):
    def __init__(self, device):
        super().__init__(device)

    @property
    def evaluation_active(self):
        return self._evaluation_active

    @evaluation_active.setter
    def evaluation_active(self, value):
        self._evaluation_active = value

    def get_point_from_output(self, out, epoch=None):
        """
        get point from encoder output
        in this case there is no exploration or other transformation
        """
        return out
