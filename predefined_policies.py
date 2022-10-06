# all policies have the form f(t),
# i. e. policy is dependence of the parameter on the current time moment and only on it
# every policy should be capable transform entire array

import numpy as np


class AbstractPolicy:
    names = None
    params = dict()

    def __call__(self, t):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.params[item]


class ConstantPolicy(AbstractPolicy):
    names = ['value']

    def __init__(self, **kwargs):
        self.params = {'value': kwargs['value']}

    def __call__(self, t):
        if isinstance(t, np.ndarray):
            return np.full_like(t, self['value'])
        return self['value']


class SinOfPowerPolicy(AbstractPolicy):
    """
    Formula: A * sin((omega * t) ** power + alpha) + bias

    """

    names = ('power', 'A', 'omega', 'alpha', 'bias')

    def __init__(self, **kwargs):
        self.params = dict()
        for name in self.names:
            self.params[name] = kwargs[name]
        assert self['alpha'] > 1e-5

    def __call__(self, t):
        return self['A'] * np.sin((self['omega'] * t) ** self['power'] + self['alpha']) + self['bias']
