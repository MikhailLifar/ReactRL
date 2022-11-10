import copy
import \
    warnings

import numpy as np

from .BaseModel import BaseModel


class LibudaModel(BaseModel):
    model_name = 'Libuda2001'

    @staticmethod
    def default_constants():
        constants = dict()

        constants['Ts'] = 440.
        constants['CTs'] = 0.3
        # # currently this parameter is in self.values dict
        # constants['S0_O2'] = 0.78 - 7.4e-4 * constants['Ts']
        constants['S0_CO'] = 0.96

        constants['thetaO_max'] = 0.25
        constants['thetaCO_max'] = 0.5

        constants['nPd'] = 1.53e+19

        # # currently this parameters are in self.values dict
        # constants['F_CO_coef'] = (2 * 3.1415926 * 28e-26 / 6.02 * 1.38e-23 * constants['Ts']) ** (-0.5)  # current
        # (2 * 3.14 * 28e-23 / 6.02 * 1.38e-23 * constants['Ts']) ** (-0.5) others
        # constants['F_O2_coef'] = (2 * 3.1415926 * 32e-26 / 6.02 * 1.38e-23 * constants['Ts']) ** (-0.5)  # current
        # (2 * 3.14 * 32e-23 / 6.02 * 1.38e-23 * constants['Ts']) ** (-0.5) others

        constants['v2'] = 1e+15
        constants['v4'] = 1e+8 * (10 ** (-0.1))  # 1e+7, 1e+8, !1e+8 * (10 ** (-0.1)) others
        constants['E2'] = 136  # 134, !136
        constants['E4'] = 60  # 62, !60, 59 others
        constants['k_B'] = 0.008314463

        # in exp is always met: p_CO + p_O2 == p_lim
        constants['p_lim'] = 1e-4  # 1e-4 Pa, according to paper

        # # currently this parameters are in self.values dict
        # constants['k1_coef'] = constants['F_CO_coef'] / constants['nPd']
        # constants['k2'] = constants['v2'] * np.exp(-constants['E2'] / constants['k_B'] / constants['Ts'])
        # constants['k3_coef'] = constants['F_O2_coef'] / constants['nPd']
        # constants['k4'] = constants['v4'] * np.exp(-constants['E4'] / constants['k_B'] / constants['Ts'])

        # limitations
        constants['O2_bottom'] = 0.
        constants['CO_bottom'] = 0.
        constants['O2_top'] = 10.e-5
        constants['CO_top'] = 10.e-5

        return constants

    def __init__(self, init_cond: dict = None,
                 **kws):
        BaseModel.__init__(self)
        self.constants = self.default_constants()
        self.values = dict()
        self.assign_constants(**kws)
        LibudaModel.new_values(self)

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
        # save initial cond
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
        self.top['output']['CO2'] = self['k4'] * self['thetaO_max'] * self['thetaCO_max']

        self.fill_limits()

        self.add_info = LibudaModel.get_add_info(self)

    def __getitem__(self, item):
        if item in self.constants:
            return self.constants[item]
        else:
            return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value

    def get_add_info(self):
        s = f'Model name: {self.model_name}\n' + ('-' * 10) + '\n'
        for name in self.constants:
            s += f'{name}: {self[name]}\n'
        return s

    def assign_constants(self, **kw):
        # change default
        for name in kw:
            if name in self.constants:
                self.constants[name] = kw[name]
                # special cases
                if (name == 'thetaCO_max') and ('thetaO_max' not in kw):
                    self.constants['thetaO_max'] = (1. - kw[name]) / 2
                    warnings.warn('thetaO_max was chosen automatically')
                if (name == 'O2_top') and ('CO_top' not in kw):
                    self.constants['O2_top'] = kw[name]
                    warnings.warn('CO_top was chosen automatically')
            else:
                raise ValueError(f'Error! Invalid constant name: {name}')

    def assign_and_eval_values(self, **kw):
        self.assign_constants(**kw)
        self.new_values()
        self.add_info = self.get_add_info()

    def new_values(self):
        Ts = self['Ts']
        # Ts = 300
        self['S0_O2'] = 0.78 - 7.4e-4 * Ts
        self['F_CO_coef'] = (2 * np.pi * 28e-26 / 6.02 * 1.38e-23 * Ts) ** (-0.5)
        self['F_O2_coef'] = (2 * np.pi * 32e-26 / 6.02 * 1.38e-23 * Ts) ** (-0.5)

        self['k1_coef'] = self['F_CO_coef'] / self['nPd']
        self['k2'] = self['v2'] * np.exp(-self['E2'] / self['k_B'] / Ts)
        self['k3_coef'] = self['F_O2_coef'] / self['nPd']
        self['k4'] = self['v4'] * np.exp(-self['E4'] / self['k_B'] / Ts)

        # LIMITATIONS ASSIGNMENT
        for name in self.names['input']:
            self.bottom['input'][name] = self[f'{name}_bottom']
            self.top['input'][name] = self[f'{name}_top']  # normal conditions
            self.fill_limits()

    def change_temperature(self, Ts: float):
        # TODO crutch here
        self.constants['Ts'] = Ts
        self.new_values()

    def update(self, data_slice, delta_t, save_for_plot=False):
        O2_p = data_slice[0]
        CO_p = data_slice[1]

        # estimation values of k:
        # k1_coef: 1500
        # k2: 0.04
        # k3_coef: 1500
        # k4: 5

        k1 = self['k1_coef'] * CO_p
        k2 = self['k2']
        k3 = self['k3_coef'] * O2_p
        k4 = self['k4']

        thetaCO = self.thetaCO
        thetaO = self.thetaO

        S_multiplier_CO = 1 - thetaCO / self['thetaCO_max'] - self['CTs'] * thetaO / self['thetaO_max']
        S_CO = self['S0_CO'] * S_multiplier_CO
        # S_CO = max(S_CO, 0)  # optional statement. Doesn't accord Libuda2001 article

        S_multiplier_O2 = 1 - thetaCO / self['thetaCO_max'] - thetaO / self['thetaO_max']
        if S_multiplier_O2 > 0:
            S_O2 = self['S0_O2'] * S_multiplier_O2 * S_multiplier_O2
        else:
            S_O2 = 0.

        self.thetaCO += (k1 * S_CO - k2 * thetaCO - k4 * thetaCO * thetaO) * delta_t
        self.thetaO += (2 * k3 * S_O2 - k4 * thetaCO * thetaO) * delta_t

        # self.plot['k1*S_CO'] = k1 * S_CO
        # self.plot['k2*thetaCO'] = k2 * thetaCO
        # self.plot['k4*thetaCO*thetaO'] = k4 * thetaCO * thetaO

        # this code makes me doubt...
        self.thetaCO = min(max(0., self.thetaCO), self['thetaCO_max'])
        self.thetaO = min(max(0., self.thetaO), self['thetaO_max'])
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['thetaCO'] = self.thetaCO
            self.plot['thetaO'] = self.thetaO

        self.model_output = k4 * self.thetaO * self.thetaCO
        self.t += delta_t
        return self.model_output

    def reset(self):
        BaseModel.reset(self)
        self.thetaCO = self.init_thetaCO
        self.thetaO = self.init_thetaO
        # self.plot = {'thetaCO': self.thetaCO * 0.01, 'thetaO': self.thetaO * 0.01}

    # measures manipulations
    def pressure_norm_func(self, gas_name):
        gas_inds = {'O2': 3, 'CO': 1}

        def norm(pressure):
            return pressure * self[f'k{gas_inds[gas_name]}_coef']

        return norm

    def pressure_to_F_value(self, pressure, gas_name):
        return self[f'F_{gas_name}_coef'] * pressure

    def CO2_rate_to_F_value(self, rate):
        return self[f'nPd'] * rate


class LibudaModelReturnK3K1(LibudaModel):
    model_name = 'LibudaReturnK3K1'

    def __init__(self, **kwargs):
        LibudaModel.__init__(self, **kwargs)
        self.names['output'] = ['CO2', 'O2(k3)', 'CO(k1)']
        for name in self.names['output']:
            self.bottom['output'][name] = 0.
        self.top['output']['CO2'] = self['k4'] * self['thetaO_max'] * self['thetaCO_max']
        self.top['output']['O2(k3)'] = self['k3_coef'] * self.top['input']['O2']
        self.top['output']['CO(k1)'] = self['k1_coef'] * self.top['input']['CO']
        self.fill_limits()

    def new_values(self):
        LibudaModel.new_values(self)
        self.top['output']['CO2'] = self['k4'] * self['thetaO_max'] * self['thetaCO_max']
        self.top['output']['O2(k3)'] = self['k3_coef'] * self.top['input']['O2']
        self.top['output']['CO(k1)'] = self['k1_coef'] * self.top['input']['CO']
        self.fill_limits()

    def update(self, data_slice, delta_t, save_for_plot=False):
        O2_p = data_slice[0]
        CO_p = data_slice[1]

        k1 = self['k1_coef'] * CO_p
        k2 = self['k2']
        k3 = self['k3_coef'] * O2_p
        k4 = self['k4']

        thetaCO = self.thetaCO
        thetaO = self.thetaO

        S_multiplier_CO = 1 - thetaCO / self['thetaCO_max'] - self['CTs'] * thetaO / self['thetaO_max']
        S_multiplier_O2 = 1 - thetaCO / self['thetaCO_max'] - thetaO / self['thetaO_max']
        S_CO = self['S0_CO'] * S_multiplier_CO

        if S_multiplier_O2 > 0:
            S_O2 = self['S0_O2'] * S_multiplier_O2 * S_multiplier_O2
        else:
            S_O2 = 0

        self.thetaCO += (k1 * S_CO - k2 * thetaCO - k4 * thetaCO * thetaO) * delta_t
        self.thetaO += (2 * k3 * S_O2 - k4 * thetaCO * thetaO) * delta_t

        # this code makes me doubt...
        self.thetaCO = min(max(0, self.thetaCO), self['thetaCO_max'])
        self.thetaO = min(max(0, self.thetaO), self['thetaO_max'])
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['thetaCO'] = self.thetaCO
            self.plot['thetaO'] = self.thetaO

        # CHANGES COMES HERE
        CO2_flow = k4 * self.thetaO * self.thetaCO
        self.model_output = np.array([CO2_flow, k3, k1])

        self.t += delta_t
        return self.model_output


class LibudaModelWithDegradation(LibudaModel):
    model_name = 'LibudaDegrad'

    def __init__(self, Ts=440., init_cond: dict = None,
                 **kwargs):
        kwargs_to_super = copy.deepcopy(kwargs)
        for name in ['v_d', 'v_r', 'border']:
            if name in kwargs_to_super:
                del kwargs_to_super[name]
        LibudaModel.__init__(self, Ts=Ts, init_cond=init_cond,
                             **kwargs_to_super)

        # degradation related attributes
        if not len(kwargs):
            # default
            self.v_degrad = 0.01
            self.v_recov = 1.5
            # self.ratio_border = 0.25
            self.ratio_border = 0.2
        else:
            self.v_degrad = kwargs['v_d']
            self.v_recov = kwargs['v_r']
            self.ratio_border = kwargs['border']
        self.D = 1.

        self.add_info = self.get_add_info()

        self.plot['D'] = self.D

    def assign_constants(self, **kw):
        if 'v_d' in kw:
            self.v_degrad = kw['v_d']
            del kw['v_d']
        if 'v_r' in kw:
            self.v_recov = kw['v_r']
            del kw['v_r']
        if 'border' in kw:
            self.ratio_border = kw['border']
            del kw['border']
        LibudaModel.assign_constants(self, **kw)

    def get_add_info(self):
        s = LibudaModel.get_add_info(self)
        s = s + f'--model with degradation--\n' + \
                f'-degradation params-\n' + \
                f'v_degrad: {self.v_degrad}\n' + \
                f'v_recov: {self.v_recov}\n' + \
                f'y_border: {self.ratio_border}\n'
        return s

    def update(self, data_slice, delta_t, save_for_plot=False):
        # update and compute output such as in the model without degradation
        LibudaModel.update(self, data_slice, delta_t, save_for_plot)
        O2 = data_slice[0]
        CO = data_slice[1]

        # # deactivation like the old one used during the tests for the russian article
        # # deactivation parameters for the tests: Vd = 0.01, Vr = 1.5, border = 4.
        # # other parameters: Ts = 440, init_cond = {'thetaO': 0.25, 'thetaCO': 0.5}
        if O2 <= CO / 2 / self.ratio_border:  # if O2 flow is too small
            ratio = 2 * self.ratio_border
        else:
            ratio = CO / O2

        # # more physical deactivation
        # if CO <= O2 / 2 / self.ratio_border:  # if CO flow is too small
        #     ratio = 2 * self.ratio_border
        # else:
        #     ratio = O2 / CO

        ratio_border = self.ratio_border
        if ratio >= ratio_border:
            # i.e. if O2(or CO) > ratio_border * CO(or O2),
            # i.e. O2(CO) flow is high enough in comparison with CO(O2) flow
            # reactivation
            self.D += self.v_recov * (1 - self.D) * (ratio - ratio_border) * delta_t
        else:
            # if O2(CO) flow is low enough in comparison with CO(O2) flow
            # deactivation
            self.D += self.v_degrad * self.D * (ratio - ratio_border) * delta_t

        self.D = max(min(self.D, 1.), 0.)  # in case of D is out of range, return D in range

        if save_for_plot:
            self.plot['D'] = self.D

        # out (of model) = out * k
        self.model_output *= self.D
        return self.model_output

    def reset(self):
        LibudaModel.reset(self)
        self.D = 1.
        self.plot['D'] = self.D


class LibudaModel_Parametrize(LibudaModel):
    # parameteres:
    # p1 -> k1_coef, p2(T) -> k2,
    # p3 -> k3_coef, p4(T)-> k4

    # TODO very big crutch with this limits!
    limits = dict()
    limits['input'] = dict()
    limits['input']['O2'] = np.array([0., 1.5 / 50 * 100000])
    limits['input']['CO'] = np.array([0., 1.5 / 50 * 100000])

    limits['out'] = dict()

    def __init__(self, Ts=440., params=None, init_cond: dict = None):
        LibudaModel.__init__(self, Ts=Ts, init_cond=init_cond)

        self.T_reference = 440.

        if params is not None:
            self.set_params(params)

    def set_params(self, params):
        BaseModel.set_params(self, params)
        if params is not None:
            self['k1_coef'] = params['p1']
            self['k2'] = params['p2'] * self['Ts'] / self.T_reference
            self['k3_coef'] = params['p3']
            self['k4'] = params['p4'] * self['Ts'] / self.T_reference

    def new_values(self):
        Ts = self['Ts']
        self['S0_O2'] = 0.78 - 7.4e-4 * Ts

        self['k2'] = self.params['p2'] * self['Ts'] / self.T_reference
        self['k4'] = self.params['p4'] * self['Ts'] / self.T_reference


class LibudaDegrad_Parametrize(LibudaModelWithDegradation):
    # parameteres:
    # p1 -> k1_coef, p2(T) -> k2,
    # p3 -> k3_coef, p4(T)-> k4

    # TODO very big crutch with this limits!
    limits = dict()
    limits['input'] = dict()
    limits['input']['O2'] = np.array([0., 1.5 / 50 * 100000])
    limits['input']['CO'] = np.array([0., 1.5 / 50 * 100000])

    limits['out'] = dict()

    def __init__(self, Ts=440., params=None, init_cond: dict = None):
        LibudaModelWithDegradation.__init__(self, Ts=Ts, init_cond=init_cond)

        self.T_reference = 440.

        if params is not None:
            self.set_params(params)

    def set_params(self, params):
        BaseModel.set_params(self, params)
        if params is not None:
            self['k1_coef'] = params['p1']
            self['k2'] = params['p2'] * self['Ts'] / self.T_reference
            self['k3_coef'] = params['p3']
            self['k4'] = params['p4'] * self['Ts'] / self.T_reference

            self.v_recov = params['v_recov']
            self.v_degrad = params['v_degrad']
            self.ratio_border = params['ratio_border']

    def new_values(self):
        Ts = self['Ts']
        self['S0_O2'] = 0.78 - 7.4e-4 * Ts

        self['k2'] = self.params['p2'] * self['Ts'] / self.T_reference
        self['k4'] = self.params['p4'] * self['Ts'] / self.T_reference
