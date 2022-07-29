import numpy as np


class BaseModel:
    names = dict()
    bottom = dict()
    top = dict()

    parameters_count = dict()
    each_pair_limits = dict()

    # define parameters names...
    names['input'] = []  # ATTENTION, this list should define the order
    # of the arguments in the update method
    bottom['input'] = {}
    top['input'] = {}
    # ...or parameters number (if don't have names) instead
    # with same limits for each output in this case
    parameters_count['input'] = 5
    each_pair_limits['input'] = (-5., 5.)

    # same for the outputs
    # define parameters names..
    names['output'] = []  # ATTENTION, this list should define the order
    # of the outputs in the update method
    bottom['output'] = {}
    top['output'] = {}
    # ...or parameters number (if don't have names) instead
    parameters_count['output'] = 10
    each_pair_limits['output'] = (0., 10.)

    model_name = 'TestModel'
    limits = {}

    def __init__(self, params=None):
        if isinstance(params, dict):
            self.params = params
        elif params is None:
            self.params = dict()
        else:
            raise AssertionError(f'Invalid value for params: {params}')
        self.t = 0.
        self.model_output = None
        self.add_info = ''

        self.plot = dict()

    def fill_limits(self):
        for kind in self.names:
            if self.names[kind]:
                self.limits[kind] = []
                for name in self.names[kind]:
                    self.limits[kind].append(
                        (self.bottom[kind][name], self.top[kind][name]))
            elif self.parameters_count[kind]:
                self.limits[kind] = [self.each_pair_limits[kind]] * self.parameters_count[kind]
            else:
                raise AttributeError('Something went wrong!')
            print(self.limits[kind])

    def set_params(self, params):
        assert isinstance(params, dict), 'Params is not a dictionary!'
        self.params = params

    def __getitem__(self, item):
        return self.params[item]

    def get_bounds(self, min_or_max: str, kind: str, out: str = 'array'):
        if min_or_max == 'min':
            in_pair = 0
        elif min_or_max == 'max':
            in_pair = 1
        else:
            raise ValueError('Return min or max - third option is not given')
        assert kind in ('input', 'output'), 'Return for inputs or outputs - third option is not given'
        if out == 'dict':
            assert self.names[kind], 'Do not have names for inputs'
            return {name: self.limits[kind][i][in_pair] for i, name in enumerate(self.names[kind])}
        elif out == 'array':
            return np.array([lim_pair[in_pair] for lim_pair in self.limits['input']])

    def update(self, data_slice, delta_t, save_for_plot=False):
        raise NotImplementedError

    def reset(self):
        self.t = 0.
        self.model_output = None

    # @staticmethod
    # def data_from_df_to_numpy(frame):
    #     out_arr = np.empty((frame.shape[0], 2))
    #     out_arr[:, 0] = frame['???']
    #     out_arr[:, -1] = frame['???']
    #     return out_arr

