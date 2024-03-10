class IncompatibleError(Exception):
    """
    raise when two components are used for embedding framework
    and they are not compatible
    """

    def __init__(self, current_component, missing_argument):
        self.message = (f"{current_component} object received incompatible input. Input should have {missing_argument}"
                        f" argument.")
        super().__init__(self.message)
