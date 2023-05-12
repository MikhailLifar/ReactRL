import warnings

import numpy as np
import matplotlib.pyplot as plt

from .BaseModel import *


class ZGBModel(BaseModel):
    """
    There is a surface of sites of size m x n
    Sites can be: 0 - empty, 1 - CO covered, 2 - O covered
    Each sim step a molecule appears
    With probability x it is CO, with probability 1 - x it is O2
    if CO
        random site is chosen
        if the site is empty
            site is occupied by CO (0 -> 1)
            four neighbors are checked
                if one is O (2)
                    CO2 is formed ((1, 2) -> (0, 0))
        else
            pass
    if O2
        two adjacent sites are randomly chosen chosen
        if the sites are empty
            sites are occupied by O ((0, 0) -> ((2, 2))
            all (six) neighbors are checked
                if one is CO (1)
                    CO2 is formed ((1, 2) -> (0, 0))
        else
            pass
    """

    model_name = 'ZGB_model'

    names = {'input': ['x'], 'output': ['CO2_prod_rate', 'x', 'CO2_count']}
    bottom = {'input': dict(), 'output': dict(), }
    top = {'input': dict(), 'output': dict(), }

    bottom['input']['x'], top['input']['x'] = 0., 1.
    bottom['output']['x'], top['output']['x'] = 0., 1.
    bottom['output']['CO2_count'] = 0
    bottom['output']['CO2_prod_rate'], top['output']['CO2_prod_rate'] = 0., 1.

    def __init__(self, m, n, **params):
        BaseModel.__init__(self, params)
        self.params['surface_size'] = (m, n)
        self.m, self.n = m, n
        self.surface = np.zeros((m, n), dtype=np.int8)
        self.area = self.m * self.n
        self.CO2_count = 0
        self.prev_CO2_count = 0
        self.thetaCO = self.thetaO = 0.

        self.top['output']['CO2_count'] = params['CO2_count_top']
        self.fill_limits()

        self.plot = {'thetaCO': self.thetaCO, 'thetaO': self.thetaO}

    # @staticmethod
    # def default_constants():
    #     pass
    #
    # def assign_constants(self, **kw):
    #     pass
    #
    # def assign_and_eval_values(self, **kw):
    #     pass

    def update(self, data_slice, delta_t, save_for_plot=False):
        # Note: now delta_t is discrete, it means number of steps
        if abs((int(delta_t) - delta_t) / delta_t) > 1.e-5:
            warnings.warn(f'In ZGB model delta_t is supposed to be integer, but {delta_t} was given')
        delta_t = int(delta_t)

        self.prev_CO2_count = self.CO2_count
        x = data_slice[0]
        for i in range(delta_t):
            self.step(x)

        self.thetaO = self.get_cov(2)
        self.thetaCO = self.get_cov(1)
        if save_for_plot:
            self.plot['thetaO'] = self.thetaO
            self.plot['thetaCO'] = self.thetaCO

        delta_CO2_count = self.CO2_count - self.prev_CO2_count
        self.model_output = np.array([delta_CO2_count / delta_t, x, delta_CO2_count])
        return self.model_output

    def step(self, x):
        # first site choice
        i, j = np.random.randint([0, 0], [self.m, self.n], (2, ))
        neighbors = [tup for tup in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1))
                     if (0 <= tup[0] < self.m) and (0 <= tup[1] < self.n)]
        gas = 'CO' if np.random.random() < x else 'O2'

        if gas == 'CO':
            if self.surface[i, j] == 0:
                # neighbors should be checked in random order
                np.random.shuffle(neighbors)
                self.surface[i, j] = 1
                for i_nn, j_nn in neighbors:
                    if (self.surface[i_nn, j_nn] == 2) and (self.surface[i, j] == 1):
                        self.surface[i_nn, j_nn] = self.surface[i, j] = 0
                        self.CO2_count += 1
        else:
            # adjacent site choice
            i1, j1 = neighbors[np.random.randint(len(neighbors))]
            # if both are empty
            if self.surface[i, j] + self.surface[i1, j1] == 0:
                # neighbors update
                neighbors.remove((i1, j1))
                neighbors += [tup for tup in ((i1 + 1, j1), (i1 - 1, j1), (i1, j1 + 1), (i1, j1 - 1))
                              if (0 <= tup[0] < self.m) and (0 <= tup[1] < self.n) and (tup != (i, j))]
                # neighbors should be checked in random order
                np.random.shuffle(neighbors)
                self.surface[i1, j1] = self.surface[i, j] = 2
                for i_nn, j_nn in neighbors:
                    if self.surface[i_nn, j_nn] == 1:
                        if (abs(i_nn - i) + abs(j_nn - j) == 1) and (self.surface[i, j] == 2):
                            self.surface[i_nn, j_nn] = self.surface[i, j] = 0
                            self.CO2_count += 1
                        elif (abs(i_nn - i1) + abs(j_nn - j1) == 1) and (self.surface[i1, j1] == 2):
                            self.surface[i_nn, j_nn] = self.surface[i1, j1] = 0
                            self.CO2_count += 1

        self.t += 1
        return

    def get_cov(self, site_type: int):
        return np.sum(self.surface == site_type) / self.area

    def reset(self):
        BaseModel.reset(self)
        self.surface[:, :] = 0
        self.CO2_count = 0
        self.prev_CO2_count = 0
        self.thetaCO = self.thetaO = 0.

        self.plot = {'thetaCO': self.thetaCO, 'thetaO': self.thetaO}


class ZGBkModel(ZGBModel):
    """
    ZGB model with desorption
    """

    model_name = 'ZGBk_model'

    predefined_params = {'k_crit': 0.0406}

    def __init__(self, m, n, **params):
        ZGBModel.__init__(self, m, n, **params)
        assert 'k' in params, 'Error! Desorption rate is not assigned'
        assert 0 <= params['k'] <= self.params['k_crit'], f'Error! Desorption rate is not within a range [0, {self["k_crit"]}]'

    def step(self, x):
        if np.random.random() < self.params['k']:
            # CO desorption
            i, j = np.random.randint([0, 0], [self.m, self.n], (2, ))
            if self.surface[i, j] == 1:
                self.surface[i, j] = 0
        else:
            ZGBModel.step(self, x)


