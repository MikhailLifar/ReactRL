import numpy as np
from scipy.integrate import solve_ivp

from .BaseModel import BaseModel


def get_dfdt_Basset(flows, ks):

    # @numba.njit
    def dfdt_Libuda(t, thetas: np.ndarray) -> np.ndarray:
        thetaO, thetaCO, thetaOsub = thetas
        free_sites = max(1 - thetaO - thetaCO, 0.)
        reaction_term = ks[2] * thetaO * thetaCO

        ret = np.array([ks[0] * (1 - thetaCO) * flows[1] - ks[1] * thetaCO - reaction_term,
                        ks[3] * flows[0] * np.exp(-ks[4] * thetaOsub) * free_sites * free_sites - reaction_term \
                            - ks[5] * thetaO * (1 - thetaOsub) + ks[6] * thetaOsub * (1 - thetaO),
                        ks[5] * thetaO * (1 - thetaOsub) - ks[6] * thetaOsub * (1 - thetaO)])

        return ret

    return dfdt_Libuda


class BassettModel(BaseModel):

    names = {'input': ['O2', 'CO'], 'output': ['CO2', 'O2', 'CO']}

    bottom = {'input': dict(), 'output': dict()}
    top = {'input': dict(), 'output': dict()}

    bottom['input']['O2'] = bottom['input']['CO'] = 0.
    top['input']['O2'] = 1.
    top['input']['CO'] = 1.

    bottom['output']['O2'] = bottom['output']['CO'] = 0.
    top['output']['O2'] = 1.
    top['output']['CO'] = 1.

    bottom['output']['CO2'] = 0.
    top['output']['CO2'] = None

    model_name = 'Bassett'

    predefined_params = {
        'rate_ads_O2': 7.6e+5,
        'rate_ads_CO': 4.1e+5,
        'rate_des_CO': 3.,
        'rate_react': 1.8e+2,
        'rate_surf_to_bulk': 9.8e-2,
        'rate_bulk_to_surf': 1.1e-3,
        'exp_factor': 10.,
    }

    def __init__(self, params=None, resample_when_reset=False, set_Libuda=False):
        # self._check_params(params)
        BaseModel.__init__(self, params)
        # self.set_Bassett()

        # initial conditions
        self.thetaO = self.thetaCO = self.thetaOsub = 0.

        # save initial cond
        self.plot = {'thetaO': self.thetaO, 'thetaCO': self.thetaCO, 'thetaOsub': self.thetaOsub, 'error': 0.}

        self.top['output']['CO2'] = 0.33 * self['rate_react']
        self.fill_limits()

    def update(self, data_slice, delta_t, save_for_plot=False):

        O2, CO = data_slice
        O2, CO = O2 * 1.e-1, CO * 1.e-3  # TODO: crutch here to simplify plotting

        ks = np.array([self['rate_ads_CO'], self['rate_des_CO'], self['rate_react'],
                       self['rate_ads_O2'], self['exp_factor'],
                       self['rate_surf_to_bulk'], self['rate_bulk_to_surf']])
        thetas = np.array([self.thetaO, self.thetaCO, self.thetaOsub])

        # NUMBA ATTEMPT
        # _, thetas, error = RK45vec_step(get_dfdt_Libuda(np.array([O2, CO]), ks, theta_max_vec, Cs), 0, thetas, delta_t)

        thetas = solve_ivp(get_dfdt_Basset(data_slice, ks), [0., delta_t], y0=thetas,
                           t_eval=[0, delta_t], atol=1.e-6, rtol=1.e-4, first_step=delta_t / 3,)

        self.thetaO, self.thetaCO, self.thetaOsub = thetas.y[:, -1]

        # this code makes me doubt...
        # self.thetaO = min(max(0., self.thetaO), 1.)
        # self.thetaCO = min(max(0., self.thetaCO), 1.)
        # self.thetaOsub = min(max(0., self.thetaOsub), 1.)
        # but it didn't influence much on results

        if save_for_plot:
            self.plot['thetaO'] = self.thetaO
            self.plot['thetaCO'] = self.thetaCO
            self.plot['thetaOsub'] = self.thetaOsub
        # self.plot['error'] = error
        self.plot['error'] = -1.

        # model_output is normalized to be between 0 and 1
        # self.model_output = np.array([(self.thetaB / self['thetaB_max']) * (self.thetaA / self['thetaA_max']), O2, CO])
        self.model_output = np.array([self.params['rate_react'] * self.thetaO * self.thetaCO, O2, CO])
        self.t += delta_t
        return self.model_output

    def reset(self):
        BaseModel.reset(self)
        self.plot = {'thetaO': self.thetaO, 'thetaCO': self.thetaCO, 'thetaOsub': self.thetaOsub, 'error': 0.}

