import numpy as np


# targets, catalytic CO oxidation
def CO2_value(x):
    # x = [CO2(k4 * thetaCO * thetaO)]
    return x[0]


def CO2xConversion(x):
    # x = [CO2(k4 * thetaCO * thetaO), O2(k3), CO(k1)]
    # formula, roughly: cost = CO2 * (1 - O2_out/O2_in - CO_out/CO_in)
    # formula was rewriting, considering O2_out = O2_in - 2 * CO2; CO_out = CO_in - CO2;
    # and using protection from division by zero
    eps = 5e-2
    return x[0] * (2 * x[0] / (x[1] + eps) + x[0] / (x[2] + eps))


def CO2_sub_outs(x):
    # x = [CO2(k4 * thetaCO * thetaO), O2(k3), CO(k1)]
    # formula: CO2 - alpha(O2_out + CO_out)
    alpha = 0.1
    return x[0] - alpha * ((x[1] - 2 * x[0]) + (x[2] - x[0]))


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
