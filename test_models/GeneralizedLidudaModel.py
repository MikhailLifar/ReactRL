import copy

import numpy as np
from scipy.integrate import solve_ivp

from .BaseModel import BaseModel

# import sys
# sys.path.append('../')

# from .solve_de import *


def get_dfdt_Libuda(flows,
                    ks,
                    theta_max_vec,
                    Cs):

    # @numba.njit
    def dfdt_Libuda(t, thetas: np.ndarray) -> np.ndarray:

        reaction_term = ks[4] * thetas[0] * thetas[1]

        StickA = 1 - thetas[1] / theta_max_vec[1] - Cs[0] * thetas[0] / theta_max_vec[0]
        # StickA = max(StickA, 0)  # optional statement. Doesn't accord Libuda2001 article
        StickB = 1 - thetas[0] / theta_max_vec[0] - Cs[1] * thetas[1] / theta_max_vec[1]
        if StickB > 0.:
            StickB *= StickB
        else:
            StickB = 0

        ret = np.array([2 * ks[2] * flows[0] * StickB - 2 * ks[3] * thetas[0] * thetas[0] - reaction_term,
                        ks[0] * flows[1] * StickA - ks[1] * thetas[1] - reaction_term])

        return ret

    return dfdt_Libuda


class GeneralizedLibudaModel(BaseModel):
    """

    The idea behind the class is to generalize Libuda model.
    This means particularly use the system of differential equations of the same form as in Libuda,
    but with different coefficients
    Since now we chose coefficients arbitrary we don't need to utilize any physical knowledge anymore

    The original system:
    d(thetaCO) / dt = k1 * S_CO - k2 * thetaCO - k4 * thetaCO * thetaO,
    d(thetaO) / dt = 2 * k3 * S_O2 * Heaviside(S_O2) - k4 * thetaCO * thetaO
    Estimates of original system coefficients:
        k1 * S0_CO: 0.1489509291107995; k2: 0.07162483150136743; k3 * S0_O2: 0.06594992906062133; k4: 5.987340360331895

    The generalized system:
    StickA = 1 - thetaA / thetaA_max - C_B_inhibit_A * thetaB / thetaB_max,  C_A_B in [0, 1]
    StickB - analogous
    d(thetaA) / dt = rate_ads_A * inputA * StickA - rate_des_A * thetaA - rate_react * thetaA * thetaB,
    d(thetaB) / dt = rate_ads_B * inputB * StickB ** 2 * Heaviside(StickB) - rate_react * thetaA * thetaB

    Parameters:
         rate_ads_A, rate_des_A, rate_ads_B, rate_react in [0.01, 10];
         thetaA_max, thetaB_max in [0.25, 1.];
         C_A_B, C_B_A in [0., 1.]
         BUT The idea to just sample from continuous range may not be so good
         Because some situations may be rare
         It should be better to choice randomly from predefined grid of parameters
    Overall 8 parameters

    """

    names = {'input': ['inputB', 'inputA'], 'output': ['outputC', 'B', 'A']}

    bottom = {'input': dict(), 'output': dict()}
    top = {'input': dict(), 'output': dict()}
    bottom['input']['inputA'] = bottom['input']['inputB'] = 0.
    top['input']['inputA'] = top['input']['inputB'] = 1.

    bottom['output']['A'] = bottom['output']['B'] = 0.
    bottom['output']['outputC'] = 0.

    top['output']['A'] = top['output']['B'] = 1.
    top['output']['outputC'] = 1.

    model_name = 'GeneralizedLibuda'

    params_names = ['thetaB_init', 'thetaA_init',
                    'thetaB_max', 'thetaA_max',
                    'rate_ads_A', 'rate_des_A', 'rate_ads_B', 'rate_des_B', 'rate_react',
                    'C_B_inhibit_A', 'C_A_inhibit_B']

    params_choices = {
        'rate': (0.01, 0.05, 0.07,
                 0.1, 0.2, 0.3,
                 0.5, 0.7, 1.,
                 2., 5., 10),
        'theta': (0.25, 0.5),
        'C': (0., 0.05, 0.1, 0.3, 0.5, 0.7, 1.),
    }

    def __init__(self, params=None, resample_when_reset=False, set_Libuda=False):
        # self._check_params(params)
        BaseModel.__init__(self, params)
        if set_Libuda:
            self.set_Libuda()
        elif params is None:
            self._sample_model()

        # initial conditions
        self.thetaB, self.thetaA = self.params['thetaB_init'], self.params['thetaA_init']
        for name in ('thetaB_init', 'thetaA_init'):
            assert self.params[name] <= self.params[f'{name[:name.find("_")]}_max'], 'Wrong assignment to initial conditions'

        # save initial cond
        self.plot = {'thetaA': self.thetaA, 'thetaB': self.thetaB, 'error': 0.}

        # self.top['output']['outputC'] = self['rate_react'] * self['thetaA_max'] * self['thetaB_max']
        self.fill_limits()

        self.resample_when_reset = resample_when_reset

    def _check_params(self, params_dict):
        for name in self.params_names:
            assert name in params_dict, f'Error! Parameter {name} is not defined'

    def _sample_model(self):
        for prefix in self.params_choices:
            for name in filter(lambda p: p.startswith(prefix), self.params_names):
                self.params[name] = np.random.choice(self.params_choices[prefix])
        for name in 'A', 'B':
            self.params[f'theta{name}_init'] = min(self.params[f'theta{name}_init'],
                                                   self.params[f'theta{name}_max'])

    def _resample(self):
        self._sample_model()

    def set_Libuda(self):
        self.set_params(
            params={
                    'thetaA_max': 0.5, 'thetaB_max': 0.25,
                    'thetaA_init': 0., 'thetaB_init': 0.25,
                    'rate_ads_A': 0.14895,
                    'rate_des_A': 0.07162,
                    'rate_ads_B': 0.06594,
                    'rate_des_B': 0.,
                    'rate_react': 5.98734,
                    'C_B_inhibit_A': 0.3,
                    'C_A_inhibit_B': 1.,
                    }
        )

    def update(self, data_slice, delta_t, save_for_plot=False):

        inputB, inputA = data_slice

        # ORIGINAL

        # k_1 = self['rate_ads_A']
        # k_des1 = self['rate_des_A']
        # k_2 = self['rate_ads_B']
        # k_des2 = self['rate_des_B']
        # k_3 = self['rate_react']

        # def step(thetaB, thetaA, dt):
        #     reaction_term = k_3 * thetaB * thetaA
        #
        #     StickA = 1 - thetaA / self['thetaA_max'] - self['C_B_inhibit_A'] * thetaB / self['thetaB_max']
        #     # StickA = max(StickA, 0)  # optional statement. Doesn't accord Libuda2001 article
        #     StickB = 1 - thetaB / self['thetaB_max'] - self['C_A_inhibit_B'] * thetaA / self['thetaA_max']
        #     StickB = (StickB * StickB) if StickB > 0. else 0.
        #
        #     thetaB_new = thetaB + (2 * k_2 * inputB * StickB - 2 * k_des2 * thetaB * thetaB - reaction_term) * dt
        #     thetaA_new = thetaA + (k_1 * inputA * StickA - k_des1 * thetaA - reaction_term) * dt
        #
        #     return thetaB_new, thetaA_new

        # def do_steps(thetaB, thetaA, dt, n_steps):
        #     for i in range(n_steps):
        #         thetaB, thetaA = step(thetaB, thetaA, dt / n_steps)
        #     return thetaB, thetaA

        # # coefs estimation print
        # print(f'k1: {self["rate_ads_A"]}; k2: {self["rate_des_A"]}; k3: {self["rate_ads_B"]}; k4: {self["rate_react"]}')

        # print(self.thetaB)
        # print(self.thetaA)
        # exit(0)

        # theta_B_1step, theta_A_1step = do_steps(self.thetaB, self.thetaA, delta_t, 1)
        # theta_B_2step, theta_A_2step = do_steps(self.thetaB, self.thetaA, delta_t, 2)
        # error = max(abs(theta_A_1step - theta_A_2step), abs(theta_B_1step - theta_B_2step))

        # self.thetaB = theta_B_2step
        # self.thetaA = theta_A_2step

        ks = np.array([self.params['rate_ads_A'], self['rate_des_A'],
                       self.params['rate_ads_B'], self['rate_des_B'],
                       self['rate_react']])
        theta_max_vec = np.array([self.params['thetaB_max'], self.params['thetaA_max']])
        Cs = np.array([self.params['C_B_inhibit_A'], self.params['C_A_inhibit_B']])
        thetas = np.array([self.thetaB, self.thetaA])

        # NUMBA ATTEMPT
        # _, thetas, error = RK45vec_step(get_dfdt_Libuda(np.array([inputB, inputA]), ks, theta_max_vec, Cs), 0, thetas, delta_t)

        thetas = solve_ivp(get_dfdt_Libuda(data_slice, ks, theta_max_vec, Cs), [0., delta_t], y0=thetas,
                           t_eval=[0, delta_t], atol=1.e-6, rtol=1.e-4, first_step=delta_t / 3,)

        self.thetaB, self.thetaA = thetas.y[:, -1]

        # this code makes me doubt...
        self.thetaB = min(max(0., self.thetaB), self['thetaB_max'])
        self.thetaA = min(max(0., self.thetaA), self['thetaA_max'])
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['thetaB'] = self.thetaB
            self.plot['thetaA'] = self.thetaA
        # self.plot['error'] = error
        self.plot['error'] = -1.

        # model_output is normalized to be between 0 and 1
        self.model_output = np.array([(self.thetaB / self['thetaB_max']) * (self.thetaA / self['thetaA_max']), inputB, inputA])
        self.t += delta_t
        return self.model_output

    def reset(self):
        BaseModel.reset(self)
        if self.resample_when_reset:
            self._resample()
        self.thetaA = self['thetaA_init']
        self.thetaB = self['thetaB_init']
        self.plot = {'thetaA': self.thetaA, 'thetaB': self.thetaB, 'error': 0.}

    @staticmethod
    def co_flow_part_to_pressure_part(x_co):
        return x_co * 7 / np.sqrt(8) / (np.sqrt(7) + 7 * x_co / np.sqrt(8) - np.sqrt(7) * x_co)


class LibudaGWithTemperature(GeneralizedLibudaModel):

    names = copy.deepcopy(GeneralizedLibudaModel.names)
    names['input'].append('T')
    names['output'].append('T')

    bottom = copy.deepcopy(GeneralizedLibudaModel.bottom)
    top = copy.deepcopy(GeneralizedLibudaModel.top)
    bottom['input']['T'] = bottom['output']['T'] = 400.
    top['input']['T'] = top['output']['T'] = 600.

    model_name = 'LibudaGWithT'

    params_names = copy.deepcopy(GeneralizedLibudaModel.params_names)
    for suff in ('ads_A', 'ads_B', 'des_A', 'des_B', 'react'):
        params_names.append([f'rate_{suff}_0', f'E_{suff}'])
        params_names.remove(f'rate_{suff}')

    predefined_params = {'kB': 0.008314463}

    def __init__(self, params=None, T=440., resample_when_reset=False, set_Libuda=False):
        self.T = T
        GeneralizedLibudaModel.__init__(self, params, resample_when_reset, set_Libuda)

    def calc_for_T(self, T):
        for suff in ('ads_A', 'ads_B', 'des_A', 'des_B', 'react'):
            self.params[f'rate_{suff}'] = self.params[f'rate_{suff}_0'] * np.exp(-self.params[f'E_{suff}'] / self['kB'] / T)

    def set_Libuda(self):

        self.set_params(
            params={
                    'thetaA_max': 0.5, 'thetaB_max': 0.25,
                    'thetaA_init': 0., 'thetaB_init': 0.25,
                    'rate_ads_A_0': 1.,
                    'rate_des_A_0': 1.,
                    'rate_ads_B_0': 1.,
                    'rate_des_B_0': 0.,
                    'rate_react_0': 1.,
                    'E_ads_A': 100.,  # chosen randomly, actual parameters should be find later
                    'E_des_A': 136.,
                    'E_ads_B': 120.,
                    'E_des_B': 0.,  # zero because in the original Libuda model there is no oxygen adsorption
                    'E_react': 60.,
                    'C_B_inhibit_A': 0.3,
                    'C_A_inhibit_B': 1.,
                    })

        self.T = 440.
        self.calc_for_T(self.T)

        k_1 = 0.14895 / self.params['rate_ads_A']
        k_des1 = 0.07162 / self.params['rate_des_A']
        k_2 = 0.06594 / self.params['rate_ads_B']
        k_des2 = 0.
        k_3 = 5.98734 / self.params['rate_react']

        self.params.update({
            'rate_ads_A_0': k_1, 'rate_des_A_0': k_des1,
            'rate_ads_B_0': k_2, 'rate_des_B_0': k_des2,
            'rate_react_0': k_3,
        })

        self.calc_for_T(self.T)

    def update(self, data_slice, delta_t, save_for_plot=False):

        inputB, inputA, T = data_slice
        if abs(self.T - T) > 1:
            self.T = T
            self.calc_for_T(self.T)

        GeneralizedLibudaModel.update(self, data_slice[:-1], delta_t, save_for_plot)

        self.model_output = np.hstack((self.model_output, [T]))
        return self.model_output


class LynchModel(BaseModel):
    """
    Model is based on the 5 dimensionless diff. equations
    3 for gases (O2, CO, CO2), 2 for coverages
    model parameters:
    K1, K1_des, K2_2, K2_des, K_3, alpha

    also:
    F_CO2 (always 0)
    F_CO and F_O2 (inputs) (F_CO + F_O2 = 1)

    default params (params from the article)
    K1: 35, K1_des: -0.5, K2: 20, K2_des: 50, K3: 6, alpha: 1

    """

    names = {'input': ['O2', 'CO'], 'output': ['CO2', 'O2', 'CO']}

    bottom = {'input': dict(), 'output': dict()}
    top = {'input': dict(), 'output': dict()}

    bottom['input']['O2'] = bottom['input']['CO'] = 0.
    top['input']['O2'] = top['input']['CO'] = 1.

    bottom['output']['O2'] = bottom['output']['CO'] = 0.
    bottom['output']['CO2'] = 0.

    top['output']['O2'] = top['output']['CO'] = 1.

    predefined_params = {'K1': 35, 'K1_des': 0.5, 'K2': 20, 'K2_des': 50, 'K3': 6,
                         'alpha': 1, 'F_CO2': 0.}

    def __init__(self, **params):
        BaseModel.__init__(self, params)
        self.X = self.Y = self.Z = 0.
        self.thetaCO = self.thetaO = 0.

        self.top['output']['CO2'] = self['K3']
        self.fill_limits()

        self.plot = {'thetaCO': self.thetaCO, 'thetaO': self.thetaO}
        self.model_output = np.empty(3)

    def update(self, data_slice, delta_t, save_for_plot=False):
        # Euler's method
        F_O2, F_CO = data_slice

        X, Y, Z = self.X, self.Y, self.Z
        thetaO, thetaCO = self.thetaO, self.thetaCO

        free_frac = (1. - thetaO - thetaCO)

        CO_adsorp_term = self.params['K1'] * X * free_frac
        CO_desorp_term = self.params['K1_des'] * thetaCO
        O2_adsorp_term = self.params['K2'] * Y * free_frac * free_frac
        O2_desorp_term = self.params['K2_des'] * thetaO * thetaO
        reaction_term = self.params['K3'] * thetaO * thetaCO

        Q_expr = 1 - CO_adsorp_term + CO_desorp_term - O2_adsorp_term + O2_desorp_term + reaction_term

        self.X += (F_CO - Q_expr * X - CO_adsorp_term + CO_desorp_term) * delta_t
        self.Y += (F_O2 - Q_expr * Y - O2_adsorp_term + O2_desorp_term) * delta_t
        self.Z += (self.params['F_CO2'] - Q_expr * Z + reaction_term) * delta_t

        self.thetaO += (2 * O2_adsorp_term - 2 * O2_desorp_term - reaction_term) * self.params['alpha'] * delta_t
        self.thetaCO += (CO_adsorp_term - CO_desorp_term - reaction_term) * self.params['alpha'] * delta_t

        if save_for_plot:
            self.plot['thetaCO'] = self.thetaCO
            self.plot['thetaO'] = self.thetaO

        self.model_output[:] = np.array([Z, F_O2, F_CO])
        # self.t += delta_t
        return self.model_output

    def reset(self):
        BaseModel.reset(self)
        self.X = self.Y = self.Z = 0.
        self.thetaCO = self.thetaO = 0.
