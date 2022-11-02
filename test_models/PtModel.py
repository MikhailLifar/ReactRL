import warnings

import numpy as np

from .BaseModel import BaseModel
from .LibudaModel import LibudaModel


class PtModel(LibudaModel):
    model_name = 'Pt_2210'

    @staticmethod
    def default_constants():
        constants = dict()

        constants['Ts'] = 440.

        constants['S_CO'] = 0.84
        constants['F_CO'] = 1.919e3
        constants['S_O2'] = 0.06
        constants['F_O2'] = 3.589e3

        constants['thetaO_max'] = 0.25
        constants['thetaCO_max'] = 0.5

        # constants['nPt'] = ???

        constants['v2'] = 1.25e+15
        constants['v4'] = 1.645e+14
        constants['E2'] = 146
        constants['E4'] = 100.8
        constants['k_B'] = 0.008314463

        # # in exp is always met: p_CO + p_O2 == p_lim
        # constants['p_lim'] = 1e-4  # 1e-4 Pa, according to paper

        constants['k1'] = constants['S_CO'] * constants['F_CO']
        constants['k2'] = constants['v2'] * np.exp(-constants['E2'] / constants['k_B'] / constants['Ts'])
        constants['k3'] = constants['S_O2'] * constants['F_O2']
        constants['k4'] = constants['v4'] * np.exp(-constants['E4'] / constants['k_B'] / constants['Ts'])

        # limitations
        constants['O2_bottom'] = 0.
        constants['CO_bottom'] = 0.
        constants['O2_top'] = 10.
        constants['CO_top'] = 10.

        return constants

    def __init__(self, init_cond: dict = None,
                 **kws):
        warnings.warn('WARNING! CRUTCH WITH THE FLOWS IN UPDATE METHOD')

        BaseModel.__init__(self)
        self.constants = self.default_constants()
        self.values = dict()
        self.assign_constants(**kws)
        PtModel.new_values(self)

        # initial conditions
        if init_cond is None:
            self.theta_CO = 0
            self.theta_O = self.constants['thetaO_max']
        else:
            for name in init_cond:
                assert init_cond[name] <= self.constants[f'{name}_max'],\
                    'This initial conditions are not allowed'
            self.theta_CO = init_cond['thetaCO']
            self.theta_O = init_cond['thetaO']
        # save initial conds
        self.init_thetaCO = self.theta_CO
        self.init_thetaO = self.theta_O
        # self.plot = {'k1*S_CO': None, 'k2*theta_CO': None, 'k4*theta_CO*theta_O': None}
        self.plot = {'theta_CO': self.theta_CO, 'theta_O': self.theta_O}

        # LIMITATIONS ASSIGNMENT
        self.names['input'] = ['O2', 'CO']
        self.bottom['input']['O2'] = self['O2_bottom']
        # self.top['input']['O2'] = 2.e-5  # limited conditions (figure in the article),
        # # were tested with deactivation
        self.top['input']['O2'] = self['O2_top']  # normal conditions
        self.bottom['input']['CO'] = self['CO_bottom']
        # self.top['input']['CO'] = 30.e-5  # CO can be much above O2
        # # to test without deactivation
        self.top['input']['CO'] = self['CO_top']  # normal conditions

        self.names['output'] = ['CO2']
        self.bottom['output']['CO2'] = 0.
        self.top['output']['CO2'] = self['k4'] * self['thetaO_max'] * self['thetaCO_max'] * self['S_CO']

        self.fill_limits()

        self.plot = {'theta_CO': self.theta_CO, 'theta_O': self.theta_O}

        self.add_info = LibudaModel.get_add_info(self)

    def assign_and_eval_values(self, **kw):
        self.assign_constants(**kw)
        self.new_values()
        self.add_info = self.get_add_info()

    def new_values(self):
        # LIMITATIONS ASSIGNMENT
        for name in self.names['input']:
            self.bottom['input'][name] = self[f'{name}_bottom']
            self.top['input'][name] = self[f'{name}_top']  # normal conditions
            self.fill_limits()

    def change_temperature(self, Ts: float):
        raise NotImplementedError

    def update(self, data_slice, delta_t, save_for_plot=False):
        O2_p = data_slice[0]
        CO_p = data_slice[1]

        # estimation values of k:
        # k1:  1.6e3
        # k2:  6e-3
        # k3:  2.2e2
        # k4:  1.8e2

        # TODO: CRUTCH HERE
        O2_p = O2_p * 1e-5
        CO_p = CO_p * 1e-5

        k1 = self['k1']
        k2 = self['k2']
        k3 = self['k3']
        k4 = self['k4']

        theta_CO = self.theta_CO
        theta_O = self.theta_O

        sticking_coef_CO = theta_CO / self['thetaCO_max']
        sticking_coef_CO = 1 - sticking_coef_CO * sticking_coef_CO
        # if sticking_coef_CO < 0:
        #     sticking_coef_CO = 0
        self.theta_CO += (k1 * CO_p * sticking_coef_CO - k2 * theta_CO - k4 * theta_CO * theta_O) * delta_t
        sticking_coef_O2 = 1 - (theta_CO / self['thetaCO_max']) - (theta_O / self['thetaO_max'])
        if sticking_coef_O2 > 0:
            sticking_coef_O2 *= sticking_coef_O2
        else:
            sticking_coef_O2 = 0
        self.theta_O += (k3 * O2_p * sticking_coef_O2 - k4 * theta_CO * theta_O) * delta_t

        # self.plot['k1*S_CO'] = k1 * S_CO
        # self.plot['k2*theta_CO'] = k2 * theta_CO
        # self.plot['k4*theta_CO*theta_O'] = k4 * theta_CO * theta_O

        # this code makes me doubt...
        self.theta_CO = min(max(0, self.theta_CO), self['thetaCO_max'])
        self.theta_O = min(max(0, self.theta_O), self['thetaO_max'])
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['theta_CO'] = self.theta_CO
            self.plot['theta_O'] = self.theta_O

        self.model_output = k4 * self.theta_O * self.theta_CO
        self.t += delta_t
        return self.model_output

    # measures manipulations
    def pressure_norm_func(self, gas_name):
        raise NotImplementedError

    def pressure_to_F_value(self, pressure, gas_name):
        raise NotImplementedError

    def CO2_rate_to_F_value(self, rate):
        raise NotImplementedError


class PtReturnK1K3Model(PtModel):
    def __init__(self, **kwargs):
        PtModel.__init__(self, **kwargs)
        self.names['output'] = ['CO2', 'O2(k3xO2)', 'CO(k1xCO)']
        for name in self.names['output']:
            self.bottom['output'][name] = 0.
        self.top['output']['CO2'] = self['k4'] * self['thetaO_max'] * self['thetaCO_max']
        self.top['output']['O2(k3xO2)'] = self['k3'] * self.top['input']['O2']
        self.top['output']['CO(k1xCO)'] = self['k1'] * self.top['input']['CO']
        self.fill_limits()

    def update(self, data_slice, delta_t, save_for_plot=False):
        O2_p = data_slice[0]
        CO_p = data_slice[1]

        # TODO: CRUTCH HERE
        O2_p = O2_p * 1e-5
        CO_p = CO_p * 1e-5

        k1 = self['k1']
        k2 = self['k2']
        k3 = self['k3']
        k4 = self['k4']

        theta_CO = self.theta_CO
        theta_O = self.theta_O

        sticking_coef_CO = theta_CO / self['thetaCO_max']
        sticking_coef_CO = 1 - sticking_coef_CO * sticking_coef_CO
        self.theta_CO += (k1 * CO_p * sticking_coef_CO - k2 * theta_CO - k4 * theta_CO * theta_O) * delta_t
        sticking_coef_O2 = 1 - (theta_CO / self['thetaCO_max']) - (theta_O / self['thetaO_max'])
        if sticking_coef_O2 > 0:
            sticking_coef_O2 *= sticking_coef_O2
        else:
            sticking_coef_O2 = 0
        self.theta_O += (k3 * O2_p * sticking_coef_O2 - k4 * theta_CO * theta_O) * delta_t

        # this code makes me doubt...
        self.theta_CO = min(max(0, self.theta_CO), self['thetaCO_max'])
        self.theta_O = min(max(0, self.theta_O), self['thetaO_max'])
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['theta_CO'] = self.theta_CO
            self.plot['theta_O'] = self.theta_O

        CO2_out = k4 * self.theta_O * self.theta_CO

        self.model_output = np.array([CO2_out, k3 * O2_p, k1 * CO_p])

        self.t += delta_t
        return self.model_output

