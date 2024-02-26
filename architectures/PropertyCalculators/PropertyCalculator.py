

class PropertyCalculator:
    def __init__(self, device):
        self.device = device

    def calculate_high_dim_property(self, train_data_loader):
        """
        compute all properties required for comparing high dimensional points
        :param train_data_loader: pytorch dataloader
        """
        return

    def get_high_dim_property(self, ind1, ind2):
        """
        get the high dimensional property from saved values
        :param ind1: index of first high dimensional point
        :param ind2: index of second high dimensional point
        """
        return

    def get_low_dim_property(self, p1, p2):
        """
        calculate low dimensional property
        :param p1: first point as pytorch Tensor
        :param p2: second point as pytorch Tensor
        """
        return
