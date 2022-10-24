# all policies have the form f(t),
# i. e. policy is dependence of the parameter on the current time moment and only on it
# every policy should be capable transform entire array

import numpy as np


class AbstractPolicy:
    names = None

    def __init__(self, params_dict):
        self.params = dict()
        self.limitations = None
        if len(params_dict) < 1:
            return
        for name in self.names:
            self.params[name] = params_dict[name]

    def _call(self, t):
        raise NotImplementedError

    def __call__(self, t):
        res = self._call(t)
        if self.limitations is not None:
            res[res < self.limitations[0]] = self.limitations[0]
            res[res > self.limitations[1]] = self.limitations[1]
        return res

    def __getitem__(self, item):
        return self.params[item]

    def set_policy(self, params):
        self.params = params

    def set_limitations(self, *args):
        self.limitations = args


class ConstantPolicy(AbstractPolicy):
    names = ('value', )

    def _call(self, t):
        if isinstance(t, np.ndarray):
            return np.full_like(t, self['value'])
        return self['value']


class TwoStepPolicy(AbstractPolicy):
    names = ('1', '2', 't1', 't2')

    def _call(self, t):
        if isinstance(t, np.ndarray):
            rems = np.floor(t / (self['t1'] + self['t2']) + 1e-5)
            rems = t - rems * (self['t1'] + self['t2'])
            res = np.full_like(t, self['2'])
            res[rems < self['t1']] = self['1']
            return res
        raise ValueError


class SinPolicy(AbstractPolicy):
    """
    Formula: A * sin(omega * t + alpha) + bias

    """

    names = ('A', 'omega', 'alpha', 'bias')

    def __init__(self, params_dict):
        AbstractPolicy.__init__(self, params_dict)
        # assert self['alpha'] > 1e-5

    def _call(self, t):
        return self['A'] * np.sin(self['omega'] * t + self['alpha']) + self['bias']


class SinOfPowerPolicy(AbstractPolicy):
    """
    Formula: A * sin((omega * t) ** power + alpha) + bias

    """

    names = ('power', 'A', 'omega', 'alpha', 'bias')

    def __init__(self, params_dict):
        AbstractPolicy.__init__(self, params_dict)
        # assert self['alpha'] > 1e-5

    def _call(self, t):
        return self['A'] * np.sin((self['omega'] * t) ** self['power'] + self['alpha']) + self['bias']
