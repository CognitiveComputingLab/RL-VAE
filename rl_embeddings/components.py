"""
main Component class
all components inherit from this class
"""
from rl_embeddings.errors import IncompatibleError


class Component:
    def __init__(self):
        self._required_inputs = []

    def check_required_input(self, **kwargs):
        """
        check if all required inputs are present in the keyword arguments
        """
        for kwa in self._required_inputs:
            if kwa not in kwargs:
                raise IncompatibleError(self, kwa)
