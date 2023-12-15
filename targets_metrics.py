import numpy as np

from lib import integral


# targets, catalytic CO oxidation


def get_target_func(func_name, **kwargs):
    """

    :param func_name:
    Valid names:
         'coordinate', 'CO2xConversion', 'CO2_sub_outs',
         'CO2_value_I', 'CO2_sub_outs_I', 'CO2xConversion_I',
         'CO2xExpOuts_I', 'CO2xCOConversion_I',
    :param kwargs:
    :return:
    """

    # SHORT TERM (ONE-ROW) TARGETS
    if func_name == 'coordinate':
        idx = kwargs.get('idx', 0)

        def coordinate(x):
            # x = [CO2(k4 * thetaCO * thetaO)]
            return x[idx]

        return coordinate

    elif func_name == 'CO2xConversion':
        eps = kwargs['eps']

        def CO2xConversion(x):
            # x = [CO2(k4 * thetaCO * thetaO), O2(k3), CO(k1)]
            # formula, roughly: cost = CO2 * (1 - O2_out/O2_in - CO_out/CO_in)
            # formula was rewriting, considering O2_out = O2_in - 2 * CO2; CO_out = CO_in - CO2;
            # and using protection from division by zero
            return x[0] * (2 * x[0] / (x[1] + eps) + x[0] / (x[2] + eps))

        return CO2xConversion

    elif func_name == 'CO2_sub_outs':
        alpha = kwargs['alpha']  # 0.1 empirically

        def CO2_sub_outs(x):
            # x = [CO2(k4 * thetaCO * thetaO), O2(k3), CO(k1)]
            # formula: CO2 - alpha(O2_out + CO_out)
            return x[0] - alpha * ((x[1] - 2 * x[0]) + (x[2] - x[0]))

        return CO2_sub_outs

    elif func_name == 'CO2_count':

        def CO2_count(x):
            # x = ['CO2_rate', 'O2_Pa', 'CO_Pa', 'CO2_count']
            return x[3]

        return CO2_count

    # LONG TERM (EPISODE) TARGETS
    elif func_name == 'CO2_value_I':

        def CO2_value_I(output_history_dt, output_history):
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            return I_CO2

        return CO2_value_I

    elif func_name == 'CO2_sub_outs_I':
        alpha = kwargs['alpha']  # 0.1 empirically

        def CO2_sub_outs_I(output_history_dt, output_history):
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            I_O2 = integral(output_history_dt, output_history[:, 1])
            I_CO = integral(output_history_dt, output_history[:, 2])
            return I_CO2 - alpha * ((I_O2 - 2 * I_CO2) + (I_CO - I_CO2))

        return CO2_sub_outs_I

    elif func_name == 'CO2xConversion_I':
        eps = kwargs['eps']  # 1. empirically

        def CO2xConversion_I(output_history_dt, output_history):
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            I_O2 = integral(output_history_dt, output_history[:, 1])
            I_CO = integral(output_history_dt, output_history[:, 2])
            # was not 2 * I_O2, but O2
            return I_CO2 * I_CO2 / (I_CO + 2 * I_O2 + eps)

        return CO2xConversion_I

    elif func_name == 'CO2xExpOuts_I':
        alpha = kwargs['alpha']

        def CO2xExpOuts_I(output_history_dt, output_history):
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            I_O2 = integral(output_history_dt, output_history[:, 1])
            I_CO = integral(output_history_dt, output_history[:, 2])
            return I_CO2 * np.exp(alpha * ((I_CO2 - I_CO) + (2 * I_CO2 - I_O2)))

        return CO2xExpOuts_I

    elif func_name == 'CO2xCOConversion_I':
        eps = kwargs['eps']  # try 1.e-2

        def CO2xCOConversion_I(output_history_dt, output_history):
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            I_CO = integral(output_history_dt, output_history[:, 2])
            return I_CO2 * I_CO2 / (I_CO + eps)

        return CO2xCOConversion_I

    elif func_name == 'CO2_plus_CO_conv_I':
        eps, alpha, beta = kwargs['eps'], kwargs['alpha'], kwargs.get('beta', 1.)

        def CO2_plus_CO_conv_I(output_history_dt, output_history):
            # x = [CO2(k4 * thetaCO * thetaO), O2(k3), CO(k1)]
            # formula, roughly: cost = (1 - alpha) * int(CO2) / (int(CO2) + eps) + alpha * int(CO2)
            # alpha = 0 <=> consider conversion only; alpha = 1 <=> consider integral (rate) only
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            I_CO = integral(output_history_dt, output_history[:, 2])
            episode_time = integral(output_history_dt, np.ones_like(output_history_dt))  # needed for normalization
            return alpha * I_CO2 / (I_CO + eps) * episode_time + (1 - alpha) * beta * I_CO2

        return CO2_plus_CO_conv_I

    # LONG-TERM, WITH GAUSS
    elif func_name == '(Gauss)x(Conv)x(Conv)_I':
        CO_0 = kwargs['default']
        sigma = kwargs['sigma']  # try 0.1 * CO_0
        eps = kwargs['eps']  # try 1.e-4

        def Gauss_CO_sub_default_x_Conv_I(output_history_dt, output_history):
            I_CO = integral(output_history_dt, output_history[:, 2])
            I_CO_press = integral(output_history_dt, output_history[:, 4])
            I_O2 = integral(output_history_dt, output_history[:, 1])
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            diff = (I_CO_press - CO_0 * output_history_dt[output_history_dt > 0.][-1])
            return np.exp((-diff) * diff / sigma) * (I_CO2 / (I_CO + eps)) * (I_CO2 / 2 / (I_O2 + eps))

        return Gauss_CO_sub_default_x_Conv_I

    elif func_name == '(Gauss)x(Conv+Conv)_I':
        CO_0 = kwargs['default']
        sigma = kwargs['sigma']  # try 0.1 * CO_0
        eps = kwargs['eps']  # try 1.e-4

        def Gauss_CO_sub_default_x_ConvSum_I(output_history_dt, output_history):
            I_CO = integral(output_history_dt, output_history[:, 2])
            I_CO_press = integral(output_history_dt, output_history[:, 4])
            I_O2 = integral(output_history_dt, output_history[:, 1])
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            diff = (I_CO_press - CO_0 * output_history_dt[output_history_dt > 0.][-1])
            return np.exp((-diff) * diff / sigma) * ((I_CO2 / (I_CO + eps)) + (I_CO2 / 2 / (I_O2 + eps)))

        return Gauss_CO_sub_default_x_ConvSum_I

    elif func_name == '(Gauss)x(Conv+alpha)x(Conv+alpha)_I':
        CO_0 = kwargs['default']
        sigma = kwargs['sigma']  # try 0.1 * CO_0
        alpha = kwargs['alpha']  # try 0.2, 0.5
        eps = kwargs['eps']  # try 1.e-4

        def Gauss_CO_sub_default_x_ConvPlusAlpha_I(output_history_dt, output_history):
            I_CO = integral(output_history_dt, output_history[:, 2])
            I_CO_press = integral(output_history_dt, output_history[:, 4])
            I_O2 = integral(output_history_dt, output_history[:, 1])
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            diff = (I_CO_press - CO_0 * output_history_dt[output_history_dt > 0.][-1])
            return np.exp((-diff) * diff / sigma) * (I_CO2 / (I_CO + eps) + alpha) * (I_CO2 / 2 / (I_O2 + eps) + alpha)

        return Gauss_CO_sub_default_x_ConvPlusAlpha_I

    elif func_name == '(Gauss)xCO2_I':
        CO_0 = kwargs['default']
        sigma = kwargs['sigma']  # try 0.1 * CO_0

        def Gauss_CO_sub_default_x_CO2_I(output_history_dt, output_history):
            I_CO_press = integral(output_history_dt, output_history[:, 4])
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            diff = (I_CO_press - CO_0 * output_history_dt[output_history_dt > 0.][-1])
            return np.exp((-diff) * diff / sigma) * I_CO2

        return Gauss_CO_sub_default_x_CO2_I

    elif func_name == '(Gauss)xCO_conv':
        CO_0 = kwargs['default']
        sigma = kwargs['sigma']  # try 0.1 * CO_0
        eps = kwargs['eps']

        # Lynch model! 3 outputs!
        def Gauss_CO_sub_default_x_CO_conv(output_history_dt, output_history):
            I_CO = integral(output_history_dt, output_history[:, 2])
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            t_last = output_history_dt[output_history_dt > 0.][-1]
            diff = (I_CO - CO_0 * t_last)
            gauss = np.exp((-diff) * diff / sigma)
            return max(gauss, 1.e-7) * (I_CO2 / (I_CO + eps)) * t_last  # TODO: crutch here

        return Gauss_CO_sub_default_x_CO_conv

    else:
        raise ValueError


# metrics, catalytic CO oxidation
def CO2_integral(output_history_dt, output_history):
    return integral(output_history_dt, output_history[:, 0])


def CO2_count(output_history_dt, output_history):
    return np.sum(output_history[:, 3])


def overall_O2_conversion(output_history_dt, output_history):
    I_CO2 = integral(output_history_dt, output_history[:, 0])
    I_O2 = integral(output_history_dt, output_history[:, 1])
    return I_CO2 / (2 * I_O2)


def overall_CO_conversion(output_history_dt, output_history):
    I_CO2 = integral(output_history_dt, output_history[:, 0])
    I_CO = integral(output_history_dt, output_history[:, 2])
    return I_CO2 / I_CO
