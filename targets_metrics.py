import numpy as np

from lib import integral


# targets, catalytic CO oxidation


def get_target_func(func_name, **kwargs):
    # SHORT TERM (ONE-ROW) TARGETS
    if func_name == 'CO2_value':

        def CO2_value(x):
            # x = [CO2(k4 * thetaCO * thetaO)]
            return x[0]

        return CO2_value

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

    # LONG TERM (EPISODE) TARGETS
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
            return I_CO2 * I_CO2 / (I_CO + I_O2 + eps)

        return CO2xConversion_I

    elif func_name == 'CO2xExpOuts_I':
        alpha = kwargs['alpha']

        def CO2xExpOuts_I(output_history_dt, output_history):
            I_CO2 = integral(output_history_dt, output_history[:, 0])
            I_O2 = integral(output_history_dt, output_history[:, 1])
            I_CO = integral(output_history_dt, output_history[:, 2])
            return I_CO2 * np.exp(alpha * ((I_CO2 - I_CO) + (2 * I_CO2 - I_O2)))

        return CO2xExpOuts_I


# metrics
def CO2_integral(output_history):
    return np.sum(output_history[:, 0])


def overall_O2_conversion(output_history):
    total_CO2 = np.sum(output_history[:, 0])
    total_O2 = np.sum(output_history[:, 1])
    return total_CO2 / (2 * total_O2)


def overall_CO_conversion(output_history):
    total_CO2 = np.sum(output_history[:, 0])
    total_CO = np.sum(output_history[:, 2])
    return total_CO2 / total_CO
