import numpy as np


class BaseModel:
    input_names = ['O2', 'CO']  # ATTENTION, this list should define the order
    # of the arguments in the update method
    model_name = '???'
    limits = {'input': dict(), 'out': dict()}

    def __init__(self, params=None):
        if isinstance(params, dict):
            self.params = params
        elif params is None:
            self.params = dict()
        else:
            raise AssertionError(f'Invalid value for params: {params}')
        self.t = 0.
        self.model_output = 0.
        self.add_info = ''

        self.plot = dict()

    def set_params(self, params):
        assert isinstance(params, dict), 'Params is not a dictionary!'
        self.params = params

    def __getitem__(self, item):
        return self.params[item]

    def max_inputs(self, out: str = 'dict'):
        if out == 'dict':
            return {name: self.limits['input'][name][1] for name in self.limits['input']}
        elif out == 'array':
            return np.array([self.limits['input'][name][1] for name in self.input_names])

    def min_inputs(self, out: str = 'dict'):
        if out == 'dict':
            return {name: self.limits['input'][name][0] for name in self.limits['input']}
        elif out == 'array':
            return np.array([self.limits['input'][name][0] for name in self.limits['input']])

    def update(self, data_slice, delta_t, save_for_plot=False):
        raise NotImplementedError

    def reset(self):
        self.t = 0.
        self.model_output = 0.

    # @staticmethod
    # def data_from_df_to_numpy(frame):
    #     out_arr = np.empty((frame.shape[0], 2))
    #     out_arr[:, 0] = frame['O2_flow']
    #     out_arr[:, -1] = frame['model_time']
    #     return out_arr

