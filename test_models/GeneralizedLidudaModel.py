import numpy as np

from .BaseModel import BaseModel


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
                    'rate_ads_A', 'rate_des_A', 'rate_ads_B', 'rate_react',
                    'C_B_inhibit_A', 'C_A_inhibit_B']

    params_choices = {
        'rate': (0.01, 0.05, 0.07,
                 0.1, 0.2, 0.3,
                 0.5, 0.7, 1.,
                 2., 5., 10),
        'theta': (0.25, 0.5),
        'C': (0., 0.05, 0.1, 0.3, 0.5, 0.7, 1.),
    }

    def __init__(self, params=None, resample_when_reset=False):
        BaseModel.__init__(self, params)
        if params is None:
            self._sample_model()

        # initial conditions
        self.thetaB, self.thetaA = self.params['thetaB_init'], self.params['thetaA_init']
        for name in ('thetaB_init', 'thetaA_init'):
            assert self.params[name] <= self.params[f'{name[:name.find("_")]}_max'], 'Wrong assignment to initial conditions'

        # save initial cond
        self.plot = {'thetaA': self.thetaA, 'thetaB': self.thetaB}

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

    # def assign_constants(self, **kw):
    #     # change default
    #     for name in kw:
    #         if name in self.constants:
    #             self.constants[name] = kw[name]
    #             # special cases
    #             if (name == 'thetaCO_max') and ('thetaO_max' not in kw):
    #                 self.constants['thetaO_max'] = (1. - kw[name]) / 2
    #                 warnings.warn('thetaO_max was chosen automatically')
    #             if (name == 'O2_top') and ('CO_top' not in kw):
    #                 self.constants['O2_top'] = kw[name]
    #                 warnings.warn('CO_top was chosen automatically')
    #         else:
    #             raise ValueError(f'Error! Invalid constant name: {name}')

    # def assign_and_eval_values(self, **kw):
    #     self.assign_constants(**kw)
    #     self.new_values()
    #     self.add_info = self.get_add_info()

    def update(self, data_slice, delta_t, save_for_plot=False):
        inputB = data_slice[0]
        inputA = data_slice[1]

        thetaA = self.thetaA
        thetaB = self.thetaB

        StickA = 1 - thetaA / self['thetaA_max'] - self['C_B_inhibit_A'] * thetaB / self['thetaB_max']
        # StickA = max(StickA, 0)  # optional statement. Doesn't accord Libuda2001 article
        StickB = 1 - thetaB / self['thetaB_max'] - self['C_A_inhibit_B'] * thetaA / self['thetaA_max']
        StickB = (StickB * StickB) if StickB > 0. else 0.

        # # coefs estimation print
        # print(f'k1: {self["rate_ads_A"]}; k2: {self["rate_des_A"]}; k3: {self["rate_ads_B"]}; k4: {self["rate_react"]}')

        self.thetaA += (self['rate_ads_A'] * inputA * StickA - self['rate_des_A'] * thetaA - self['rate_react'] * thetaA * thetaB) * delta_t
        self.thetaB += (2 * self['rate_ads_B'] * inputB * StickB - self['rate_react'] * thetaA * thetaB) * delta_t

        # this code makes me doubt...
        self.thetaA = min(max(0., self.thetaA), self['thetaA_max'])
        self.thetaB = min(max(0., self.thetaB), self['thetaB_max'])
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['thetaB'] = self.thetaB
            self.plot['thetaA'] = self.thetaA

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
        self.plot = {'thetaA': self.thetaA, 'thetaB': self.thetaB}


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
