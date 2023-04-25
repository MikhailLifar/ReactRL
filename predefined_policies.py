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
    #
    # def __init__(self, params, minimal_t_sum):
    #     AbstractPolicy.__init__(self, params)

    def _call(self, t):
        if isinstance(t, np.ndarray):
            rems = np.floor(t / (self['t1'] + self['t2']) + 1e-5)  # TODO bug with 1e-5
            rems = t - rems * (self['t1'] + self['t2'])
            res = np.full_like(t, self['2'])
            res[rems < self['t1']] = self['1']
            return res
        raise ValueError


class AnyStepPolicy(AbstractPolicy):
    # TODO try to finish this class
    names = ()

    def __init__(self, nsteps, params_dict):
        self.nsteps = nsteps
        self.names = tuple([str(i) for i in range(1, nsteps + 1)] + [f't{i}' for i in range(1, nsteps + 1)])
        AbstractPolicy.__init__(self, params_dict)
        # if len(params_dict):
        #     self.t_sum = np.sum([self[f't{i}'] for i in range(1, nsteps + 1)])

    def _call(self, t):
        if isinstance(t, np.ndarray):
            ts = np.array([0] + [self[f't{i}'] for i in range(1, self.nsteps + 1)])
            cum_ts = np.cumsum(ts)
            rems = np.floor(t / cum_ts[-1] + 1e-5)  # TODO bug with 1e-5
            rems = t - rems * cum_ts[-1]
            res = np.empty_like(t)
            for i, t in enumerate(cum_ts[1:]):
                res[(rems >= cum_ts[i]) & (rems < t)] = self[f'{i + 1}']
            return res
        raise ValueError


class SinPolicy(AbstractPolicy):
    """
    Formula: A * sin(omega * t + alpha) + bias

    """

    names = ('A', 'T', 'alpha', 'bias')

    def __init__(self, params_dict):
        AbstractPolicy.__init__(self, params_dict)
        # assert self['alpha'] > 1e-5

    def _call(self, t):
        return self['A'] * np.sin(2 * np.pi / self['T'] * t + self['alpha']) + self['bias']


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
