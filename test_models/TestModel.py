# import \
#     copy

import numpy as np
# from numpy.linalg import norm
from numpy.linalg import matrix_rank

from .BaseModel import BaseModel


# def are_dependent(a, b):
#     return abs(np.dot(a, b) - abs(norm(a) * norm(b))) < 1e-5


def generate_max_rank_matr(m, n, randmin=-5, randmax=6,
                           dtype: np.dtype = np.float):
    # np.random.seed(100)
    # np.random.seed(10)
    # np.random.seed(1)
    # np.random.seed(200)
    # np.random.seed(300)
    M = np.zeros((m, n), dtype=dtype)
    min_dim = min(m, n)
    basis = np.eye(min_dim)
    # first row
    current_row = np.random.randint(randmin, randmax, n)
    # first vector is no zero
    while np.all(current_row[:min_dim] == np.zeros(min_dim)):
        current_row = np.random.randint(randmin, randmax, n)
    M[0] = current_row
    for i in range(1, min_dim):
        current_row = np.random.randint(randmin, randmax, n)
        matr_to_check = np.vstack((M[:i, :min_dim], current_row[:min_dim]))
        if matrix_rank(matr_to_check) < i + 1:
            print('In condition')
            for i_basis, v in enumerate(basis):
                matr_to_check = np.vstack((M[:i, :min_dim], v))
                print(matr_to_check)
                if matrix_rank(matr_to_check) == i + 1:
                    current_row[:min_dim] = current_row[:min_dim] + np.random.randint(1, randmax) * v
                    basis = np.vstack((basis[:i_basis], basis[i_basis + 1:]))
                    break
        M[i] = current_row
    if m > n:
        for i in range(n, m):
            M[i] = np.random.randint(randmin, randmax, n)
    assert matrix_rank(M) == min_dim, 'Something went wrong with the method!'
    return M


def normalize_matrix(M):
    """
    Function normalizes the matrix so that
    for any input vector x and for any output vector Mx:
        abs(Mx[j]) <= max(abs(x)) for all j

    :param M:
    :return:
    """
    assert len(M.shape) == 2
    M_T = M.transpose()
    for ind, row in enumerate(M_T):
        M_T[ind] = M_T[ind] / np.sum(np.abs(row))
        assert np.sum(np.abs(M_T[ind])) <= 1.
        M[:, ind] = M_T[ind]


class TestModel(BaseModel):
    model_name = 'TestModel'

    def __init__(self):
        BaseModel.__init__(self, params=None)

        # assign limits
        self.parameters_count['input'] = 3
        self.each_pair_limits['input'] = (-5., 5.)

        self.parameters_count['output'] = 3
        self.each_pair_limits['output'] = (-5., 5.)

        self.fill_limits()

        m = max(len(self.names['input']), self.parameters_count['input'])
        n = max(len(self.names['output']), self.parameters_count['output'])
        self.matrix_shape = (m, n)
        # self.internal_M = generate_max_rank_matr(*self.matrix_shape)
        self.internal_M = np.load(f'M_{m}x{n}.npy')
        normalize_matrix(self.internal_M)

        self.directions = np.random.randint(-1, 2, self.matrix_shape)
        self.increment = 0.01
        self.change_dir_prob = 0.01

    def set_params(self, params):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def update(self, input_arr, delta_t, save_for_plot=False):
        # # gradually changing internal
        # self.internal_M += self.directions * self.increment * delta_t
        # # change increment direction for small part of self.internal elements
        # choice_matr = np.random.random(self.matrix_shape) < self.change_dir_prob
        # d_change_vector = self.directions[choice_matr].reshape(1, -1)[0]
        # for i, d in enumerate(d_change_vector):
        #     if d == -1:
        #         d_change_vector[i] = 1
        #     else:
        #         d_change_vector[i] -= 1
        # self.directions[choice_matr] = d_change_vector
        # normalize_matrix(self.internal_M)
        self.model_output = np.dot(input_arr, self.internal_M)
        return self.model_output
