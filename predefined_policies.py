# all policies have the form f(t),
# i. e. policy is dependence of the parameter on the current time moment and only on it
# every policy should be capable transform entire array

import numpy as np


class AbstractPolicy:
    names = None

    def __init__(self, params_dict=None):
        self.params = dict()
        self.limitations = None
        if (params_dict is None) or (len(params_dict) < 1):
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
        r = t - (self['t1'] + self['t2']) * np.floor(t / (self['t1'] + self['t2']) + 1.e-5)
        if r <= self['t1']:
            return self['1']
        return self['2']


class AnyStepPolicy(AbstractPolicy):
    # TODO try to finish this class
    names = ()

    def __init__(self, nsteps, params_dict=None):
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
    Formula: A * sin(2 * Pi * t / T + alpha) + bias

    """

    names = ('A', 'T', 'alpha', 'bias')

    def __init__(self, params_dict=None):
        AbstractPolicy.__init__(self, params_dict)
        # assert self['alpha'] > 1e-5

    def _call(self, t):
        return self['A'] * np.sin(2 * np.pi / self['T'] * t + self['alpha']) + self['bias']


class SinOfPowerPolicy(AbstractPolicy):
    """
    Formula: A * sin((omega * t) ** power + alpha) + bias

    """

    names = ('power', 'A', 'omega', 'alpha', 'bias')

    def __init__(self, params_dict=None):
        AbstractPolicy.__init__(self, params_dict)
        # assert self['alpha'] > 1e-5

    def _call(self, t):
        return self['A'] * np.sin((self['omega'] * t) ** self['power'] + self['alpha']) + self['bias']


class FourierSeriesPolicy(AbstractPolicy):
    names = ('a_sin', 'a_cos', 'length')

    def __init__(self, nterms, params_dict=None):
        self.nterms = nterms
        AbstractPolicy.__init__(self, params_dict)

    def _call(self, t):
        if isinstance(t, np.ndarray):
            temp = np.tile(t, (self.nterms, 1)).T * np.arange(1, self.nterms + 1) * np.pi / self['length']
            return np.sum(np.sin(temp) * self['a_sin'] + np.cos(temp) * self['a_cos'], axis=1)
        temp = np.full(self.nterms, t) * np.arange(1, self.nterms + 1) * np.pi / self['length']
        return np.sum(np.sin(temp) * self['a_sin'] + np.cos(temp) * self['a_cos'])


class RandomTurnsPolicy(AbstractPolicy):
    names = ('bounds', 'period')
    default_turns_number = 20

    def __init__(self, params_dict=None):
        AbstractPolicy.__init__(self, params_dict)
        if params_dict is not None:
            self.random_turns = np.random.uniform(*(self['bounds']), self.default_turns_number)

    def set_policy(self, params):
        self.params = params
        self.reset()

    def reset(self):
        self.random_turns = np.random.uniform(*(self['bounds']), self.default_turns_number)

    def _call(self, t):
        if isinstance(t, np.ndarray):
            ret = t // self['period']
            for i in range(min(ret), max(ret)):
                while i >= len(self.random_turns):
                    self.random_turns = np.hstack((self.random_turns, np.random.uniform(*(self['bounds']),
                                                                                        len(self.random_turns))))
                ret[ret == i] = self.random_turns[i]
            return ret
        else:
            idx = int(t // self['period'])
            while idx >= len(self.random_turns):
                self.random_turns = np.hstack((self.random_turns, np.random.uniform(*(self['bounds']),
                                                                                    len(self.random_turns))))
            return self.random_turns[idx]
