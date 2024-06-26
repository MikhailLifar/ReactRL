import copy
from collections.abc import Iterable

import numpy as np
from scipy.integrate import solve_ivp
import sympy

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
        self.top['output']['outputC'] = self.params.get('reaction_rate_top', self['rate_react'])
        self.fill_limits()

        self.resample_when_reset = resample_when_reset

        self.to_calc_steady_state = None

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
            # params={
            #         'thetaA_max': 0.5, 'thetaB_max': 0.25,
            #         'thetaA_init': 0., 'thetaB_init': 0.25,
            #         'rate_ads_A': 0.18135,  # fitted 0.18135, 0.14895
            #         'rate_des_A': 0.076460,  # fitted 0.076460, 0.07162
            #         'rate_ads_B': 0.077423,  # fitted 0.077423, 0.06594
            #         'rate_des_B': 0.,
            #         'rate_react': 5.83692,  # fitted 5.83692, 5.98734
            #         'C_B_inhibit_A': 0.3,
            #         'C_A_inhibit_B': 1.,
            #         }
            params={
                    'thetaA_max': 0.5, 'thetaB_max': 0.25,
                    'thetaA_init': 0., 'thetaB_init': 0.25,

                    # 'rate_ads_A': 0.14895,
                    # 'rate_des_A': 0.07162,
                    # 'rate_ads_B': 0.06594,
                    # 'rate_des_B': 0.,
                    # 'rate_react': 5.98734,

                    'rate_ads_A': 0.17582,  # fit 3, MAPE 1 % for 14 points
                    'rate_des_A': 0.06927,
                    'rate_ads_B': 0.074495,
                    'rate_des_B': 0.,
                    'rate_react': 6.288,

                    'C_B_inhibit_A': 0.3,
                    'C_A_inhibit_B': 1.,
                    }
        )

    def assign_and_eval_values(self, **kw):
        self.top['output']['outputC'] = kw.get('reaction_rate_top', self.top['output']['outputC'])
        self.set_params(kw)
        self.fill_limits()

    def update(self, data_slice, delta_t, save_for_plot=False):

        inputB, inputA = data_slice

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
        # self.model_output = np.array([(self.thetaB / self['thetaB_max']) * (self.thetaA / self['thetaA_max']), inputB, inputA])
        self.model_output = np.array([self.params['rate_react'] * self.thetaB * self.thetaA, inputB, inputA])
        self.t += delta_t
        return self.model_output

    def reset(self):
        BaseModel.reset(self)
        if self.resample_when_reset:
            self._resample()
        self.thetaA = self['thetaA_init']
        self.thetaB = self['thetaB_init']
        self.plot = {'thetaA': self.thetaA, 'thetaB': self.thetaB, 'error': 0.}

    def rates(self):
        return {'k1': self['rate_ads_A'], 'k1_des': self['rate_des_A'],
                'k2': self['rate_ads_B'], 'k2_des': self['rate_des_B'],
                'k3': self['rate_react']}

    @staticmethod
    def sym_part_calc_steady_state():
        pA, pB = sympy.symbols('pA, pB')
        thetaB, thetaA = sympy.symbols('thetaB, thetaA')
        thB_max, thA_max, c_b_a, c_a_b = sympy.symbols('thB_max, thA_max, c_b_a, c_a_b')
        r_a_ads, r_a_des, r_b_ads, r_b_des, r_react = sympy.symbols('r_a_ads, r_a_des, r_b_ads, r_b_des, r_react')

        StickA = 1 - thetaA / thA_max - c_b_a * thetaB / thB_max
        StickB = (1 - c_a_b * thetaA / thA_max - thetaB / thB_max)
        StickB = StickB * StickB
        react_term = r_react * thetaA * thetaB

        eq1 = r_a_ads * pA * StickA - r_a_des * thetaA - react_term
        eq2 = 2 * r_b_ads * pB * StickB - 2 * r_b_des * thetaB * thetaB - react_term

        thA_through_thB = sympy.solve(eq1, thetaA)[0]
        eq2 = sympy.simplify(eq2.subs(thetaA, thA_through_thB) * (thB_max*(pA*r_a_ads + r_a_des*thA_max + r_react*thA_max*thetaB) ** 2))
        poly = sympy.Poly(eq2, thetaB)
        return poly, thA_through_thB

    def steady_state_sol(self, pB, pA):
        if self.to_calc_steady_state is None:
            self.to_calc_steady_state = self.sym_part_calc_steady_state()

        poly, thA_through_thB = self.to_calc_steady_state
        to_subs = {'thB_max': self['thetaB_max'], 'thA_max': self['thetaA_max'],
                   'c_b_a': self['C_B_inhibit_A'], 'c_a_b': self['C_A_inhibit_B'],
                   'r_a_ads': self['rate_ads_A'], 'r_a_des': self['rate_des_A'],
                   'r_b_ads': self['rate_ads_B'], 'r_b_des': self['rate_des_B'],
                   'r_react': self['rate_react'],
                   }
        thetaB = sympy.symbols('thetaB')
        this_poly = poly.subs(to_subs)
        thA_through_thB = thA_through_thB.subs(to_subs)

        if not isinstance(pB, Iterable):
            pB, pA = [pB], [pA]
        res = np.full((len(pB), 3), np.nan)

        tol = 1.e-5

        for i, pb, pa in zip(range(len(res)), pB, pA):
            trouble_flag = False
            poly_it = sympy.Poly(this_poly.subs({'pB': pb, 'pA': pa}), thetaB)
            thA_through_thB_it = thA_through_thB.subs({'pB': pb, 'pA': pa})
            all_coeffs = np.array(poly_it.all_coeffs(), dtype='float')
            if not np.any(all_coeffs):
                # solution may be undefined if there is no desorption
                res[i] = float(thA_through_thB_it.subs('thetaB', 0.))
            else:
                roots_debug = np.roots(all_coeffs)
                roots = np.roots(all_coeffs)
                roots = roots[np.abs(roots.imag) < tol].real
                roots = roots[(roots > -tol) & (roots < self['thetaB_max'] + tol)]
                if len(roots) > 1:
                    trouble_flag = np.max(np.abs(roots - roots[0])) > tol
                    if trouble_flag:
                        print('We are in the deep trouble')
                    else:
                        roots = [np.mean(roots)]
                # assert len(roots) == 1  # release

                # DEBUG
                if len(roots) < 1:
                    print('Something went wrong!')
                if len(roots) > 1:
                    print('Matters are even worse!!!')

                thetaB_sol = min(max(roots[0], 0.), self['thetaB_max'])
                thetaA_sol = float(thA_through_thB_it.subs('thetaB', thetaB_sol))

                if trouble_flag:
                    print(f'Trouble with parameters: {pb}, {pa}')
                    print(f'thetaB alternatives', list(roots))
                    print(f'thetaA alternatives', [float(thA_through_thB_it.subs('thetaB', r)) for r in roots])

                res[i] = [self['rate_react'] * thetaB_sol * thetaA_sol, thetaB_sol, thetaA_sol]

        return res

    def reverse_steady_state_problem(self, thetaB, thetaA):
        if not isinstance(thetaB, Iterable):
            thetaB = np.full((1, 1), thetaB, dtype='float')
            thetaA = np.full((1, 1), thetaA, dtype='float')

        precision = 1.e-10
        assert np.all(-precision <= thetaB) and np.all(thetaB <= self['thetaB_max'] + precision)
        assert np.all(-precision <= thetaA) and np.all(thetaA <= self['thetaA_max'] + precision)

        StickA = 1 - thetaA / self['thetaA_max'] - self['C_B_inhibit_A'] * thetaB / self['thetaB_max']
        StickB = (1 - self['C_A_inhibit_B'] * thetaA / self['thetaA_max'] - thetaB / self['thetaB_max'])
        StickB = StickB * StickB * (StickB > 0)
        react_term = self['rate_react'] * thetaA * thetaB

        pB = (2 * self['rate_des_B'] * thetaB * thetaB + react_term) / (2 * StickB * self['rate_ads_B'])
        pA = (self['rate_des_A'] * thetaA + react_term) / (StickA * self['rate_ads_A'])

        return pB, pA

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
    top['input']['T'] = top['output']['T'] = 700.

    model_name = 'LibudaGWithT'

    params_names = copy.deepcopy(GeneralizedLibudaModel.params_names)
    for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react'):
        params_names.append([f'rate_{suff}_0', f'E_{suff}'])
        params_names.remove(f'rate_{suff}')

    predefined_params = {'kB': 0.008314463}

    def __init__(self, params=None, T=440., resample_when_reset=False, set_Libuda=False):
        self.T = T
        GeneralizedLibudaModel.__init__(self, params, resample_when_reset, set_Libuda)

        # self.top['output']['outputC'] = params.get('reaction_rate_top', self.params['rate_react'])
        # self.fill_limits()

    def calc_for_T(self, T):
        for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react'):
            self.params[f'rate_{suff}'] = self.params[f'rate_{suff}_0'] * np.exp(-self.params[f'E_{suff}'] / self['kB'] / T)

    def set_Libuda(self):

        self.set_params(
            params={
                    'thetaA_max': 0.5, 'thetaB_max': 0.25,
                    'thetaA_init': 0., 'thetaB_init': 0.25,
                    'rate_ads_A_0': 1.,
                    'rate_des_A_0': 1.,
                    'rate_ads_B_0': 1.,
                    'rate_des_B_0': 0.,  # 0 because in the original Libuda model there is no oxygen desorption
                    'rate_react_0': 1.,
                    'E_ads_A': 0.,  # 0 because there is no adsorption temperature dependence in the article
                    'E_des_A': 136.,
                    'E_ads_B': 0.,
                    'E_des_B': 0.,
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


class LibudaGWithTEnergies(LibudaGWithTemperature):
    """
    The idea behind the class is to enable varying energies in the same manner as temperature
    """

    names = copy.deepcopy(LibudaGWithTemperature.names)
    names['input'] += ['E_ads_A', 'E_des_A', 'E_ads_B', 'E_des_B', 'E_react', ]
    names['output'] += ['E_ads_A', 'E_des_A', 'E_ads_B', 'E_des_B', 'E_react', ]

    bottom = copy.deepcopy(LibudaGWithTemperature.bottom)
    top = copy.deepcopy(LibudaGWithTemperature.top)
    for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react'):
        bottom['input'][f'E_{suff}'] = bottom['output'][f'E_{suff}'] = 0.
        top['input'][f'E_{suff}'] = top['output'][f'E_{suff}'] = 200.

    params_names = copy.deepcopy(GeneralizedLibudaModel.params_names)
    for suff in ('ads_A', 'des_A', 'ads_B', 'des_B', 'react'):
        params_names.append(f'rate_{suff}_0')
        params_names.remove(f'rate_{suff}')

    predefined_params = copy.deepcopy(LibudaGWithTemperature.predefined_params)

    model_name = 'LibudaGWithTEnergies'

    def __init__(self, params=None, T=440., Es=(100., 136., 120., 0., 60.), resample_when_reset=False, set_Libuda=False):
        self.Es = np.array(Es)  # order of Es is always following: ads_A, des_A, ads_B, des_B, react
        self.T = T
        self.calc_for_T = None
        LibudaGWithTemperature.__init__(self, params, T, resample_when_reset, set_Libuda)

    def _calc_for_T_and_Es(self, T, Es):
        for i, suff in enumerate(('ads_A', 'des_A', 'ads_B', 'des_B', 'react')):
            self.params[f'rate_{suff}'] = self.params[f'rate_{suff}_0'] * np.exp(-Es[i] / self['kB'] / T)

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
                    'C_B_inhibit_A': 0.3,
                    'C_A_inhibit_B': 1.,
                    })

        self.T = 440.
        self.Es = np.array((100., 136., 120., 0., 60.))
        self._calc_for_T_and_Es(self.T, self.Es)

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

        self._calc_for_T_and_Es(self.T, self.Es)

    def update(self, data_slice, delta_t, save_for_plot=False):

        inputB, inputA, T = data_slice[:3]
        Es = data_slice[3:]
        if (data_slice.size < 3) or (np.linalg.norm([T, *Es]) > 1):
            self.T = T
            self.Es = Es
            self._calc_for_T_and_Es(T, Es)

        GeneralizedLibudaModel.update(self, data_slice[:2], delta_t, save_for_plot)

        self.model_output = np.hstack((self.model_output, [T, *Es]))
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

