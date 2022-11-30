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
            self.thetaCO = 0
            self.thetaO = self.constants['thetaO_max']
        else:
            for name in init_cond:
                assert init_cond[name] <= self.constants[f'{name}_max'],\
                    'This initial conditions are not allowed'
            self.thetaCO = init_cond['thetaCO']
            self.thetaO = init_cond['thetaO']
        # save initial conds
        self.init_thetaCO = self.thetaCO
        self.init_thetaO = self.thetaO
        # self.plot = {'k1*S_CO': None, 'k2*thetaCO': None, 'k4*thetaCO*thetaO': None}
        self.plot = {'thetaCO': self.thetaCO, 'thetaO': self.thetaO}

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

        self.plot = {'thetaCO': self.thetaCO, 'thetaO': self.thetaO}

        self.add_info = LibudaModel.get_add_info(self)

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

        thetaCO = self.thetaCO
        thetaO = self.thetaO

        sticking_coef_CO = thetaCO / self['thetaCO_max']
        sticking_coef_CO = 1 - sticking_coef_CO * sticking_coef_CO
        # if sticking_coef_CO < 0:
        #     sticking_coef_CO = 0
        self.thetaCO += (k1 * CO_p * sticking_coef_CO - k2 * thetaCO - k4 * thetaCO * thetaO) * delta_t
        sticking_coef_O2 = 1 - (thetaCO / self['thetaCO_max']) - (thetaO / self['thetaO_max'])
        if sticking_coef_O2 > 0:
            sticking_coef_O2 *= sticking_coef_O2
        else:
            sticking_coef_O2 = 0
        self.thetaO += (k3 * O2_p * sticking_coef_O2 - k4 * thetaCO * thetaO) * delta_t

        # self.plot['k1*S_CO'] = k1 * S_CO
        # self.plot['k2*thetaCO'] = k2 * thetaCO
        # self.plot['k4*thetaCO*thetaO'] = k4 * thetaCO * thetaO

        # this code makes me doubt...
        self.thetaCO = min(max(0, self.thetaCO), self['thetaCO_max'])
        self.thetaO = min(max(0, self.thetaO), self['thetaO_max'])
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['thetaCO'] = self.thetaCO
            self.plot['thetaO'] = self.thetaO

        self.model_output = k4 * self.thetaO * self.thetaCO
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

        thetaCO = self.thetaCO
        thetaO = self.thetaO

        sticking_coef_CO = thetaCO / self['thetaCO_max']
        sticking_coef_CO = 1 - sticking_coef_CO * sticking_coef_CO
        self.thetaCO += (k1 * CO_p * sticking_coef_CO - k2 * thetaCO - k4 * thetaCO * thetaO) * delta_t
        sticking_coef_O2 = 1 - (thetaCO / self['thetaCO_max']) - (thetaO / self['thetaO_max'])
        if sticking_coef_O2 > 0:
            sticking_coef_O2 *= sticking_coef_O2
        else:
            sticking_coef_O2 = 0
        self.thetaO += (k3 * O2_p * sticking_coef_O2 - k4 * thetaCO * thetaO) * delta_t

        # this code makes me doubt...
        self.thetaCO = min(max(0, self.thetaCO), self['thetaCO_max'])
        self.thetaO = min(max(0, self.thetaO), self['thetaO_max'])
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['thetaCO'] = self.thetaCO
            self.plot['thetaO'] = self.thetaO

        CO2_out = k4 * self.thetaO * self.thetaCO

        self.model_output = np.array([CO2_out, k3 * O2_p, k1 * CO_p])

        self.t += delta_t
        return self.model_output


class PtSalomons(LibudaModel):
    model_name = 'PtSalomons'

    @staticmethod
    def default_constants():
        constants = dict()

        constants['S_CO'] = 0.84
        constants['M_CO'] = 0.028  # kg / mol
        constants['S_O2'] = 0.07
        constants['M_O2'] = 0.032  # kg / mol

        constants['Ts'] = 440.  # or 293, K
        # constants['CTs'] = 0.3
        # constants['S0_CO'] = 0.96
        #
        constants['thetaO_max'] = 1.
        constants['thetaOO_max'] = 1.
        constants['thetaCO_max'] = 1.
        #
        constants['nPt'] = 2.72e-5  # mol / m^-2
        #
        constants['v2'] = 1.e+13  # /s
        constants['v4'] = 1.006e+12  # /s
        constants['v5'] = 1.e+12  # /s
        constants['v6'] = 1.e+15  # /s
        constants['v7'] = 1.e+15  # /s

        constants['E2'] = 126.4  # kJ / mol
        constants['E4'] = 108  # kJ / mol
        constants['E5'] = 50  # kJ / mol
        constants['E6'] = 115  # kJ / mol
        constants['E7'] = 105  # kJ / mol

        constants['k_B'] = 0.83114463  # kJ / (K mol)
        constants['R'] = 8.31  # kg m^2 / (s^2 K mol)
        constants['alpha'] = 33  # _
        constants['dE4dCO'] = 33  # _
        #
        # constants['p_lim'] = 1e-4  # 1e-4 Pa, according to paper
        #
        # # limitations
        constants['O2_bottom'] = 0.
        constants['CO_bottom'] = 0.
        constants['O2_top'] = 62.5  # mol / m^3; 2000 ppm
        constants['CO_top'] = 71.43  # mol / m^3; 2000 ppm

        return constants

    def __init__(self, init_cond: dict = None,
                 **kws):
        BaseModel.__init__(self)
        self.constants = self.default_constants()
        self.values = dict()
        self.assign_constants(**kws)
        PtSalomons.new_values(self)

        # initial conditions
        if init_cond is None:
            self.thetaCO = 0.
            self.thetaO = 0.
            self.thetaOO = 0.
        else:
            for name in init_cond:
                assert (init_cond[name] >= 0.) and (init_cond[name] <= self[f'{name}_max']),\
                    'This initial conditions are not allowed'
            self.thetaCO = init_cond['thetaCO']
            self.thetaO = init_cond['thetaO']
            self.thetaOO = init_cond['thetaOO']
        # save initial conds
        self.init_thetaCO = self.thetaCO
        self.init_thetaO = self.thetaO
        self.init_thetaOO = self.thetaOO
        # self.plot = {'k1*S_CO': None, 'k2*thetaCO': None, 'k4*thetaCO*thetaO': None}
        self.plot = {'thetaCO': self.thetaCO, 'thetaO': self.thetaO, 'thetaOO': self.thetaOO, }

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
        self.top['output']['CO2'] = self['r4_const_part'] + self['r6_const_part']

        self.fill_limits()

        self.plot = {'thetaCO': self.thetaCO,
                     'thetaO': self.thetaO,
                     'thetaOO': self.thetaOO}

        self.add_info = LibudaModel.get_add_info(self)

    def new_values(self):
        Ts = self['Ts']

        self['r1_const_part'] = (self['S_CO'] / self['nPt']) * np.sqrt(self['R'] / (2 * np.pi * self['M_CO'])) * np.sqrt(Ts)
        self['r3_const_part'] = (self['S_O2'] / self['nPt']) * np.sqrt(self['R'] / (2 * np.pi * self['M_O2'])) * np.sqrt(Ts)

        for i in (2, 4, 5, 6, 7):
            self[f'r{i}_const_part'] = self[f'v{i}'] * np.exp(-self[f'E{i}'] / self['k_B'] / Ts)

    def update(self, data_slice, delta_t, save_for_plot=False):
        O2_c = data_slice[0]
        CO_c = data_slice[1]

        # estimation values of k:

        # # TODO: CRUTCH HERE
        # O2_c = O2_c * 1e-5
        # CO_c = CO_c * 1e-5

        thetaCO = self.thetaCO
        thetaO = self.thetaO
        thetaOO = self.thetaO

        r1 = self['r1_const_part'] * CO_c * (1 - thetaCO - thetaO - thetaOO)

        b3 = 1 - thetaCO - thetaO - thetaOO
        if b3 > 0.:
            b3 = b3 * b3
        else:
            b3 = 0.
        r3 = self['r3_const_part'] * O2_c * b3

        r2 = self['r2_const_part'] * thetaCO * np.exp(self['alpha'] * thetaCO / (self['k_B'] * self['Ts']))
        r4 = self['r4_const_part'] * thetaCO * thetaO * np.exp(self['dE4dCO'] * thetaCO / (self['k_B'] * self['Ts']))
        r5 = self['r5_const_part'] * CO_c * thetaO * thetaO
        r6 = self['r4_const_part'] * thetaCO * thetaOO
        r7 = self['r4_const_part'] * thetaOO * (1 - thetaCO - thetaO - thetaOO)

        self.thetaCO += r1 - r2 - r4 + r5 - r6
        self.thetaO += 2 * r3 - 2 * r4 - r5 + 2 * r6 + r7
        self.thetaOO += r5 - r6 - r7

        # this code makes me doubt...
        self.thetaCO = min(max(0, self.thetaCO), self['thetaCO_max'])
        self.thetaO = min(max(0, self.thetaO), self['thetaO_max'])
        self.thetaOO = min(max(0, self.thetaOO), self['thetaOO_max'])
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['thetaCO'] = self.thetaCO
            self.plot['thetaO'] = self.thetaO
            self.plot['thetaOO'] = self.thetaOO

        self.model_output = r4 + r6
        self.t += delta_t
        return self.model_output

    def change_temperature(self, Ts: float):
        raise NotImplementedError

    # measures manipulations
    def pressure_norm_func(self, gas_name):
        raise NotImplementedError

    def pressure_to_F_value(self, pressure, gas_name):
        raise NotImplementedError

    def CO2_rate_to_F_value(self, rate):
        raise NotImplementedError
