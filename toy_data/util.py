from importlib import import_module


class DynamicImporter:

    def __init__(self, name, verbose=False):
        self._name = name
        self._module = None
        self._verbose = verbose

    def _assert_import(self):
        if self._module is None:
            if self._verbose > 1:
                print(f"Importing module '{self._name}'...")
            elif self._verbose:
                print(f"Importing module '{self._name}'")
            self._module = import_module(self._name)
            if self._verbose > 1:
                print("DONE")

    def __getattribute__(self, item):
        if item in ['_module', '_name', '_assert_import', '_verbose']:
            return super().__getattribute__(item)
        else:
            self._assert_import()
            return getattr(self._module, item)
