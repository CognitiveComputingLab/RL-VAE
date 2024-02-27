from architectures.PropertyCalculators.PropertyCalculator import PropertyCalculator


class PropertyCalculatorNone(PropertyCalculator):
    def __init__(self, device, data_loader):
        super().__init__(device, data_loader)

    @property
    def high_dim_property(self):
        return

    def calculate_high_dim_property(self):
        return

    def get_high_dim_property(self, ind1, ind2):
        return

    def get_low_dim_property(self, p1, p2):
        return


